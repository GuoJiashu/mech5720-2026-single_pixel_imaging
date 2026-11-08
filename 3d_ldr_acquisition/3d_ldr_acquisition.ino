const uint8_t SENSOR_PINS[] = {A0, A1, A2, A3};
const uint8_t NUM_CHANNELS =
    sizeof(SENSOR_PINS) / sizeof(SENSOR_PINS[0]);

const unsigned long SAMPLE_INTERVAL_MS = 20;
unsigned long lastSampleMs = 0;

void setup() {
  Serial.begin(115200);
  analogReference(DEFAULT);

  for (uint8_t i = 0; i < NUM_CHANNELS; i++) {
    pinMode(SENSOR_PINS[i], INPUT);
  }
}

void loop() {
  unsigned long now = millis();

  if (now - lastSampleMs < SAMPLE_INTERVAL_MS) {
    return;
  }
  lastSampleMs = now;

  unsigned long timestampUs = micros();
  int adc[NUM_CHANNELS];

  // Read the five inputs sequentially as ADC counts.
  for (uint8_t i = 0; i < NUM_CHANNELS; i++) {
    adc[i] = analogRead(SENSOR_PINS[i]);
  }

  for (uint8_t i = 0; i < NUM_CHANNELS; i++) {
    Serial.print("  A");
    Serial.print(i);
    Serial.print(":");
    Serial.print(adc[i]);
  }

  Serial.println();
}