/*
 * BioTherma-AI - MCU layer (STM32U585 / Arduino Core on Zephyr)
 *
 * Non-blocking acquisition of:
 *   - AM2320 temperature + relative humidity over I2C (Wire, header SDA/SCL)
 *   - 10k NTC thermistor on A0 via 10k divider referenced to 3V3
 * Pushes telemetry to the Linux side over Bridge RPC.
 * Accepts LED state commands from Python.
 *
 * ELECTRICAL NOTE: every rail here is 3V3. The UNO Q headers are NOT 5V.
 * A0/A1 are ADC-only and NOT 5V tolerant (abs max 3.6V at the pin).
 */

#include <Arduino.h>
#include <Wire.h>
#include <Arduino_RouterBridge.h>
#include <math.h>

// ---------------- Pin map ----------------
static const uint8_t PIN_NTC        = A0;  // divider midpoint, 3V3 -> NTC -> A0 -> 10k -> GND
static const uint8_t PIN_ARRAY_PWR  = D2;  // PNP base via 1k; LOW = array rail on
static const uint8_t PIN_LED_GREEN  = D3;  // cathode side via 220R; LOW = lit
static const uint8_t PIN_LED_YELLOW = D4;
static const uint8_t PIN_LED_RED    = D5;

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
static uint8_t  g_ledMask  = 0;   // bit0 green, bit1 yellow, bit2 red
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

// ---------------- LED array ----------------
static void applyLeds() {
  // Array rail: PNP emitter on 3V3, base via 1k to D2. LOW = conducting.
  digitalWrite(PIN_ARRAY_PWR, g_arrayOn ? LOW : HIGH);

  // Cathodes sink into the MCU. LOW = lit, HIGH = dark.
  digitalWrite(PIN_LED_GREEN,  (g_arrayOn && (g_ledMask & 0x01)) ? LOW : HIGH);
  digitalWrite(PIN_LED_YELLOW, (g_arrayOn && (g_ledMask & 0x02)) ? LOW : HIGH);
  digitalWrite(PIN_LED_RED,    (g_arrayOn && (g_ledMask & 0x04)) ? LOW : HIGH);
}

// Called from Linux: set_status(mask, arrayOn)
static void onSetStatus(int mask, int arrayOn) {
  g_ledMask = (uint8_t)(mask & 0x07);
  g_arrayOn = (arrayOn != 0);
  applyLeds();
}

// ---------------- NTC ----------------
static float readNtcCelsius() {
  uint32_t acc = 0;
  for (uint8_t i = 0; i < 8; i++) acc += analogRead(PIN_NTC);
  float counts = (float)acc / 8.0f;
  if (counts < 1.0f || counts > ADC_FS - 1.0f) return NAN;  // open or shorted

  // NTC on the high side: V(A0) = 3V3 * Rf / (Rntc + Rf)
  float rNtc = NTC_R_FIXED * ((ADC_FS / counts) - 1.0f);
  float kelvin = 1.0f / ((1.0f / NTC_T25_K) + (1.0f / NTC_BETA) * logf(rNtc / NTC_R25));
  return kelvin - 273.15f;
}

// ---------------- AM2320 non-blocking state machine ----------------
static void am2320Poll(uint32_t now) {
  switch (amState) {

    case AM_IDLE:
      if (now - lastSample < SAMPLE_PERIOD_MS) return;
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
        return;
      }
      amState = AM_REQUEST_SENT;
      amStamp = now;
      break;
    }

    case AM_REQUEST_SENT:
      if (now - amStamp < AM2320_CONV_MS) return;
      amState = AM_READING;
      break;

    case AM_READING: {
      uint8_t buf[8];
      uint8_t n = Wire.requestFrom((uint8_t)AM2320_ADDR, (uint8_t)8);
      if (n != 8) { g_errCount++; amState = AM_IDLE; lastSample = now; return; }
      for (uint8_t i = 0; i < 8; i++) buf[i] = Wire.read();

      uint16_t rxCrc = (uint16_t)buf[7] << 8 | buf[6];
      if (buf[0] != 0x03 || buf[1] != 0x04 || crc16(buf, 6) != rxCrc) {
        g_errCount++;
        amState = AM_IDLE;
        lastSample = now;
        return;
      }

      uint16_t rawRh = ((uint16_t)buf[2] << 8) | buf[3];
      int16_t  rawT  = (int16_t)(((uint16_t)buf[4] << 8) | buf[5]);
      bool negative = rawT & 0x8000;
      if (negative) rawT &= 0x7FFF;

      g_rh    = rawRh / 10.0f;
      g_tempC = (negative ? -rawT : rawT) / 10.0f;

      amState = AM_IDLE;
      lastSample = now;
      break;
    }
  }
}

// ---------------- Bridge ----------------
static void publish() {
  // Fixed-point x100 to avoid float marshalling ambiguity across the RPC.
  int32_t t   = isnan(g_tempC) ? INT32_MIN : (int32_t)lroundf(g_tempC * 100.0f);
  int32_t rh  = isnan(g_rh)    ? INT32_MIN : (int32_t)lroundf(g_rh    * 100.0f);
  int32_t ntc = isnan(g_ntcC)  ? INT32_MIN : (int32_t)lroundf(g_ntcC  * 100.0f);
  Bridge.notify("telemetry", t, rh, ntc, (int32_t)g_errCount);
}

void setup() {
  pinMode(PIN_ARRAY_PWR,  OUTPUT);
  pinMode(PIN_LED_GREEN,  OUTPUT);
  pinMode(PIN_LED_YELLOW, OUTPUT);
  pinMode(PIN_LED_RED,    OUTPUT);
  digitalWrite(PIN_ARRAY_PWR, HIGH);   // array rail off until Linux is up
  applyLeds();

  analogReadResolution(ADC_BITS);

  Wire.begin();
  Wire.setClock(100000);               // AM2320 is 100kHz max

  Bridge.begin();
  Bridge.provide("set_status", onSetStatus);

  lastSample = millis() - SAMPLE_PERIOD_MS;
}

void loop() {
  uint32_t now = millis();

  if (now - lastNtc >= NTC_PERIOD_MS) {
    lastNtc = now;
    g_ntcC = readNtcCelsius();
  }

  Am2320State before = amState;
  am2320Poll(now);
  if (before == AM_READING && amState == AM_IDLE) {
    publish();
  }

  Bridge.update();
}
