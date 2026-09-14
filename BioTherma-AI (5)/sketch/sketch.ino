/*
 * BioTherma-AI - MCU layer (STM32U585 / Arduino Core on Zephyr)
 *
 * Acquisition of:
 *   - AM2320 temperature + relative humidity, either over the single-wire
 *     (DHT-compatible) bus on a 3-pin module, or over I2C on a 4-pin module
 *   - optional 10k NTC thermistor on A0 via a divider referenced to 3V3
 * Pushes telemetry to the Linux side over Bridge RPC.
 * Accepts status commands from Python and drives the two onboard RGB LEDs
 * that belong to the MCU (LED3 and LED4). Both are active LOW.
 *
 * ELECTRICAL NOTE: every rail here is 3V3. The UNO Q headers are NOT 5V.
 * A0/A1 are ADC-only and NOT 5V tolerant (abs max 3.6V at the pin).
 */

#include <Arduino.h>
#include <Wire.h>
#include <Arduino_RouterBridge.h>
#include <math.h>

// ---------------- Build options ----------------
// The thermistor divider needs a fixed resistor to ground; there is no way to
// synthesise one in software. With HAS_NTC 0 the analog channel is skipped
// entirely and the AM2320's own temperature reading becomes the process
// temperature. Put the sensor in the headspace, not outside the reactor.
#define HAS_NTC 0

// The STM32 internal pull-ups are roughly 40 kOhm -- weak, but adequate for one
// device on a short bus at 100 kHz. Most AM2320 breakout boards also carry
// their own pull-ups. If reads fail CRC intermittently, that is the first thing
// to suspect and real 4.7k-10k resistors are the fix.
#define USE_INTERNAL_PULLUPS 1

// The AM2320 speaks two protocols and the breakout you have decides which.
//   SENSOR_SINGLE_WIRE 1 -> 3-pin module: VCC, GND, DAT. DHT22-compatible
//                           single-bus protocol on PIN_SENSOR.
//   SENSOR_SINGLE_WIRE 0 -> 4-pin module: VCC, GND, SDA, SCL. I2C at 0x5C.
// A 3-pin module physically cannot do I2C -- two data lines are required.
#define SENSOR_SINGLE_WIRE 1

// ---------------- Pin map ----------------
static const uint8_t PIN_NTC    = A0;  // divider midpoint (only when HAS_NTC)
static const uint8_t PIN_SENSOR = D2;  // AM2320 DAT line, single-wire mode

// Single-wire needs a pull-up on the data line. The STM32 internal pull-up is
// weak but works over a short lead; a 4.7k to 3V3 is the fix if reads are
// flaky. Many 3-pin modules already have one fitted on the board.

// The UNO Q carries four onboard RGB LEDs. Two belong to the MPU; LED3 and
// LED4 belong to this microcontroller. No external LEDs, resistors or
// switching transistor are used.
//
//   LED3  process state   -- what the reactor is doing
//   LED4  health state    -- whether the readings can be trusted
//
// Both are active LOW: writing LOW lights that channel.

// ---------------- ADC / thermistor ----------------
static const uint8_t  ADC_BITS   = 12;
static const float    ADC_FS     = (float)((1UL << ADC_BITS) - 1);
static const float    NTC_R_FIXED = 10000.0f;  // lower leg, to GND
static const float    NTC_R25     = 10000.0f;
static const float    NTC_BETA    = 3950.0f;   // <-- match your actual part
static const float    NTC_T25_K   = 298.15f;

// ---------------- AM2320 ----------------
static const uint8_t  AM2320_ADDR     = 0x5C;
static const uint32_t AM2320_WAKE_US  = 1200;  // datasheet: >800us after wake NAK
static const uint32_t AM2320_CONV_MS  = 2;     // >1.5ms between request and read

// ---------------- Timing ----------------
static const uint32_t SAMPLE_PERIOD_MS = 2000;  // AM2320 must not be polled faster than 2s
static const uint32_t NTC_PERIOD_MS    = 250;

// ---------------- State ----------------
enum Am2320State : uint8_t {
  AM_IDLE,
  AM_WAKE_SENT,
  AM_REQUEST_SENT,
  AM_READING
};

static Am2320State amState = AM_IDLE;
static uint32_t    amStamp = 0;
static uint32_t    lastSample = 0;
static uint32_t    lastNtc = 0;

static float   g_tempC   = NAN;   // AM2320 air temp
static float   g_rh      = NAN;   // AM2320 relative humidity, %
static float   g_ntcC    = NAN;   // slurry/wall probe temp
static uint16_t g_errCount = 0;
static uint8_t  g_process  = 0;   // P_* enum below
static uint8_t  g_health   = 0;   // H_* enum below
static bool     g_arrayOn  = true;

// ---------------- CRC16 (Modbus) ----------------
static uint16_t crc16(const uint8_t *buf, uint8_t len) {
  uint16_t crc = 0xFFFF;
  while (len--) {
    crc ^= *buf++;
    for (uint8_t i = 0; i < 8; i++) {
      if (crc & 0x0001) { crc >>= 1; crc ^= 0xA001; }
      else              { crc >>= 1; }
    }
  }
  return crc;
}

// ---------------- Status LEDs ----------------
// Colour is a 3-bit mask: bit0 red, bit1 green, bit2 blue.
static const uint8_t C_OFF     = 0x00;
static const uint8_t C_RED     = 0x01;
static const uint8_t C_GREEN   = 0x02;
static const uint8_t C_AMBER   = 0x03;  // red + green
static const uint8_t C_BLUE    = 0x04;
static const uint8_t C_MAGENTA = 0x05;
static const uint8_t C_CYAN    = 0x06;
static const uint8_t C_WHITE   = 0x07;

// Process states, as sent by the Linux side.
enum : uint8_t { P_IDLE = 0, P_PRODUCING, P_BALANCE, P_HARVEST };
// Health states.
enum : uint8_t { H_IDLE = 0, H_OK, H_WATCH, H_STALL, H_BAND };

static uint8_t colourForProcess(uint8_t s) {
  switch (s) {
    case P_PRODUCING: return C_GREEN;
    case P_BALANCE:   return C_AMBER;
    case P_HARVEST:   return C_BLUE;
    default:          return C_OFF;
  }
}

static uint8_t colourForHealth(uint8_t s) {
  switch (s) {
    case H_OK:    return C_GREEN;
    case H_WATCH: return C_AMBER;
    case H_STALL: return C_RED;
    case H_BAND:  return C_MAGENTA;
    default:      return C_OFF;
  }
}

static void writeRgb(uint8_t rPin, uint8_t gPin, uint8_t bPin, uint8_t colour) {
  // Active low: LOW lights the channel.
  digitalWrite(rPin, (colour & 0x01) ? LOW : HIGH);
  digitalWrite(gPin, (colour & 0x02) ? LOW : HIGH);
  digitalWrite(bPin, (colour & 0x04) ? LOW : HIGH);
}

static void applyLeds() {
  uint8_t p = g_arrayOn ? colourForProcess(g_process) : C_OFF;
  uint8_t h = g_arrayOn ? colourForHealth(g_health)   : C_OFF;
  writeRgb(LED3_R, LED3_G, LED3_B, p);
  writeRgb(LED4_R, LED4_G, LED4_B, h);
}

// Called from Linux: set_status(process, health, enabled)
static void onSetStatus(int process, int health, int enabled) {
  g_process = (uint8_t)process;
  g_health  = (uint8_t)health;
  g_arrayOn = (enabled != 0);
  applyLeds();
}

// Called from Linux: lamp_test() -- walks both LEDs through every channel so a
// failed solder joint or a dead channel is obvious before a demo.
static void onLampTest() {
  const uint8_t seq[] = {C_RED, C_GREEN, C_BLUE, C_WHITE, C_OFF};
  for (uint8_t i = 0; i < sizeof(seq); i++) {
    writeRgb(LED3_R, LED3_G, LED3_B, seq[i]);
    writeRgb(LED4_R, LED4_G, LED4_B, seq[i]);
    delay(300);
  }
  applyLeds();
}

// ---------------- NTC ----------------
static float readNtcCelsius() {
#if !HAS_NTC
  return NAN;
#else
  uint32_t acc = 0;
  for (uint8_t i = 0; i < 8; i++) acc += analogRead(PIN_NTC);
  float counts = (float)acc / 8.0f;
  if (counts < 1.0f || counts > ADC_FS - 1.0f) return NAN;  // open or shorted

  // NTC on the high side: V(A0) = 3V3 * Rf / (Rntc + Rf)
  float rNtc = NTC_R_FIXED * ((ADC_FS / counts) - 1.0f);
  float kelvin = 1.0f / ((1.0f / NTC_T25_K) + (1.0f / NTC_BETA) * logf(rNtc / NTC_R25));
  return kelvin - 273.15f;
#endif
}

// ---------------- AM2320 single-wire (DHT-compatible) ----------------
#if SENSOR_SINGLE_WIRE

// Waits for the line to reach `level`, up to `timeout_us`. Returns the time
// spent waiting, or 0 on timeout.
static uint32_t waitLevel(bool level, uint32_t timeout_us) {
  uint32_t t0 = micros();
  while (digitalRead(PIN_SENSOR) != (level ? HIGH : LOW)) {
    if (micros() - t0 > timeout_us) return 0;
  }
  uint32_t d = micros() - t0;
  return d ? d : 1;
}

// One complete single-bus transaction. Blocks for about 5 ms, which is fine at
// a 2 s sample period. Returns true and fills the globals on success.
//
// Frame: 40 bits, MSB first, as RH_hi RH_lo T_hi T_lo checksum.
// Each bit is ~50 us low followed by a high pulse: short (~26 us) is 0,
// long (~70 us) is 1.
static bool am2320ReadSingleWire() {
  uint8_t data[5] = {0, 0, 0, 0, 0};

  // Start: hold the line low well past the 800 us minimum, then release.
  pinMode(PIN_SENSOR, OUTPUT);
  digitalWrite(PIN_SENSOR, LOW);
  delay(2);
  digitalWrite(PIN_SENSOR, HIGH);
  delayMicroseconds(30);
  pinMode(PIN_SENSOR, INPUT_PULLUP);

  // Sensor answers with ~80 us low then ~80 us high.
  if (!waitLevel(false, 200)) return false;
  if (!waitLevel(true, 200)) return false;
  if (!waitLevel(false, 200)) return false;

  for (uint8_t i = 0; i < 40; i++) {
    if (!waitLevel(true, 150)) return false;     // ~50 us start-of-bit low
    uint32_t high = waitLevel(false, 200);       // length of the high pulse
    if (!high) return false;
    data[i / 8] <<= 1;
    if (high > 45) data[i / 8] |= 1;             // threshold between 26 and 70
  }

  uint8_t sum = data[0] + data[1] + data[2] + data[3];
  if (sum != data[4]) return false;

  uint16_t rawRh = ((uint16_t)data[0] << 8) | data[1];
  uint16_t rawT = ((uint16_t)data[2] << 8) | data[3];
  bool negative = rawT & 0x8000;
  if (negative) rawT &= 0x7FFF;

  float rh = rawRh / 10.0f;
  float t = (negative ? -(float)rawT : (float)rawT) / 10.0f;

  // Reject obviously impossible frames -- a mis-timed read can still checksum.
  if (rh < 0.0f || rh > 100.0f || t < -40.0f || t > 80.0f) return false;

  g_rh = rh;
  g_tempC = t;
  return true;
}

static int8_t am2320Poll(uint32_t now) {
  if (now - lastSample < SAMPLE_PERIOD_MS) return -1;
  lastSample = now;
  if (am2320ReadSingleWire()) return 1;
  g_errCount++;
  return 0;
}

#else

static int8_t am2320Poll(uint32_t now) {
  switch (amState) {

    case AM_IDLE:
      if (now - lastSample < SAMPLE_PERIOD_MS) return -1;
      // Wake pulse: address the device and expect a NAK. Ignore the result.
      Wire.beginTransmission(AM2320_ADDR);
      Wire.endTransmission();
      delayMicroseconds(AM2320_WAKE_US);   // sub-millisecond, acceptable inline
      amState = AM_WAKE_SENT;
      amStamp = now;
      break;

    case AM_WAKE_SENT: {
      // Function 0x03, start register 0x00, read 4 bytes
      Wire.beginTransmission(AM2320_ADDR);
      Wire.write((uint8_t)0x03);
      Wire.write((uint8_t)0x00);
      Wire.write((uint8_t)0x04);
      if (Wire.endTransmission() != 0) {
        g_errCount++;
        amState = AM_IDLE;
        lastSample = now;
        return 0;
      }
      amState = AM_REQUEST_SENT;
      amStamp = now;
      break;
    }

    case AM_REQUEST_SENT:
      if (now - amStamp < AM2320_CONV_MS) return -1;
      amState = AM_READING;
      break;

    case AM_READING: {
      uint8_t buf[8];
      uint8_t n = Wire.requestFrom((uint8_t)AM2320_ADDR, (uint8_t)8);
      if (n != 8) { g_errCount++; amState = AM_IDLE; lastSample = now; return 0; }
      for (uint8_t i = 0; i < 8; i++) buf[i] = Wire.read();

      uint16_t rxCrc = (uint16_t)buf[7] << 8 | buf[6];
      if (buf[0] != 0x03 || buf[1] != 0x04 || crc16(buf, 6) != rxCrc) {
        g_errCount++;
        amState = AM_IDLE;
        lastSample = now;
        return 0;
      }

      uint16_t rawRh = ((uint16_t)buf[2] << 8) | buf[3];
      int16_t  rawT  = (int16_t)(((uint16_t)buf[4] << 8) | buf[5]);
      bool negative = rawT & 0x8000;
      if (negative) rawT &= 0x7FFF;

      g_rh    = rawRh / 10.0f;
      g_tempC = (negative ? -rawT : rawT) / 10.0f;

      amState = AM_IDLE;
      lastSample = now;
      return 1;
    }
  }
  return -1;
}

#endif  // SENSOR_SINGLE_WIRE

// ---------------- Bridge ----------------
static void publish() {
  // Fixed-point x100 to avoid float marshalling ambiguity across the RPC.
  int32_t t   = isnan(g_tempC) ? INT32_MIN : (int32_t)lroundf(g_tempC * 100.0f);
  int32_t rh  = isnan(g_rh)    ? INT32_MIN : (int32_t)lroundf(g_rh    * 100.0f);
  int32_t ntc = isnan(g_ntcC)  ? INT32_MIN : (int32_t)lroundf(g_ntcC  * 100.0f);
  Bridge.notify("telemetry", t, rh, ntc, (int32_t)g_errCount);
}

void setup() {
  const uint8_t ledPins[] = {LED3_R, LED3_G, LED3_B, LED4_R, LED4_G, LED4_B};
  for (uint8_t i = 0; i < sizeof(ledPins); i++) {
    pinMode(ledPins[i], OUTPUT);
    digitalWrite(ledPins[i], HIGH);    // active low, so HIGH is dark
  }
  applyLeds();

#if HAS_NTC
  analogReadResolution(ADC_BITS);
#endif

#if SENSOR_SINGLE_WIRE
  pinMode(PIN_SENSOR, INPUT_PULLUP);
  delay(1200);                         // the AM2320 needs ~1 s after power-up
#else
#if USE_INTERNAL_PULLUPS
  // Must come before Wire.begin(), which takes the pins over.
  pinMode(SDA, INPUT_PULLUP);
  pinMode(SCL, INPUT_PULLUP);
#endif
  Wire.begin();
  Wire.setClock(100000);               // AM2320 is 100kHz max
#endif

  Serial.begin(115200);

  Bridge.begin();
  Bridge.provide("set_status", onSetStatus);
  Bridge.provide("lamp_test", onLampTest);

  lastSample = millis() - SAMPLE_PERIOD_MS;
}

void loop() {
  uint32_t now = millis();

  if (now - lastNtc >= NTC_PERIOD_MS) {
    lastNtc = now;
    g_ntcC = readNtcCelsius();
  }

  int8_t result = am2320Poll(now);
  if (result >= 0) {
    if (result == 0) {
      g_tempC = NAN;
      g_rh = NAN;
    }
    publish();

    Serial.print("[mcu] t=");
    if (isnan(g_tempC)) Serial.print("--"); else Serial.print(g_tempC, 1);
    Serial.print("C rh=");
    if (isnan(g_rh)) Serial.print("--"); else Serial.print(g_rh, 1);
    Serial.print("% errors=");
    Serial.println(g_errCount);
  }

  Bridge.update();
}
