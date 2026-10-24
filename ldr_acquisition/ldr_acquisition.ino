#include <math.h>
#include <stdlib.h>
#include <string.h>

// Reflective Single-Pixel Imaging
// Arduino UNO R3 - Shared Acquisition
const uint8_t LDR_PIN = A0;
const uint16_t N_SAMPLES = 125;
const unsigned long SAMPLE_INTERVAL_US = 4000UL;

void performMeasurement(int id) {
  float mean = 0.0f;
  float M2 = 0.0f;
  int minADC = 1023;
  int maxADC = 0;

  unsigned long startUs = micros();
  unsigned long nextUs = startUs;

  for (uint16_t i = 0; i < N_SAMPLES; i++) {
    while ((long)(micros() - nextUs) < 0) {
    }

    int raw = analogRead(LDR_PIN);
    if (raw < minADC)
      minADC = raw;
    if (raw > maxADC)
      maxADC = raw;

    // Running mean and variance
    float delta = raw - mean;
    mean += delta / (i + 1);
    M2 += delta * (raw - mean);
    nextUs += SAMPLE_INTERVAL_US;
  }

  unsigned long captureMs = (micros() - startUs) / 1000UL;
  float stdADC = sqrt(M2 / (N_SAMPLES - 1));

  // RESULT,id,mean,std,min,max,n_samples,capture_ms
  Serial.print(F("RESULT,"));
  Serial.print(id);
  Serial.print(',');
  Serial.print(mean, 4);
  Serial.print(',');
  Serial.print(stdADC, 4);
  Serial.print(',');
  Serial.print(minADC);
  Serial.print(',');
  Serial.print(maxADC);
  Serial.print(',');
  Serial.print(N_SAMPLES);
  Serial.print(',');
  Serial.println(captureMs);
}

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(200);
  analogReference(DEFAULT);
  pinMode(LDR_PIN, INPUT);
  delay(300);
  Serial.println(F("READY,LDR_ACQUISITION"));
}

void loop() {
  if (!Serial.available())
    return;

  char buffer[32];
  size_t n = Serial.readBytesUntil('\n', buffer, sizeof(buffer) - 1);
  buffer[n] = '\0';

  if (strncmp(buffer, "MEASURE,", 8) == 0) {
    performMeasurement(atoi(buffer + 8));
  }
}
