const int trigPin = 9;
const int echoPin = 10;
const int relayPin = 8;
const int buzzerPin = 11;

void setup() {
  // Initialize serial communication
  Serial.begin(115200);

  // Initialize the ultrasonic sensor pins
  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);

  // Initialize the relay pin as an output
  pinMode(relayPin, OUTPUT);
  digitalWrite(relayPin, HIGH);

  // Initialize the buzzer pin as an output
  pinMode(buzzerPin, OUTPUT);
  digitalWrite(buzzerPin, LOW); // Ensure buzzer is off at start
}

void loop() {
  // Measure distance
  long distance = measureDistance();

  // Send distance to the serial port
  Serial.print("Distance:");
  Serial.println(distance);

  // Control relay based on distance
  if (distance <15) {
    digitalWrite(relayPin, LOW);
  } else {
// delay(8000);
     // Turn off relay
  }

  // Check for serial input and control buzzer
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    if (command.equals("buzz")) {
      playBuzzer();
    }
      if (command.equals("lightOff")) {
        // delay(3000)
digitalWrite(relayPin, HIGH);
    }
  }


  delay(500); // Wait for 0.5 seconds before next measurement
}

long measureDistance() {
  // Clear the trigPin by setting it LOW
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);

  // Set the trigPin HIGH for 10 microseconds
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);

  // Read the echoPin and calculate the duration of the pulse
  long duration = pulseIn(echoPin, HIGH);

  // Calculate the distance (in cm) based on the speed of sound
  long distance = duration * 0.034 / 2;

  return distance;
}

void playBuzzer() {
digitalWrite(relayPin, HIGH);

  // Turn the buzzer on
  digitalWrite(buzzerPin, HIGH);
  delay(2000); // Buzzer on for 1 second
  digitalWrite(buzzerPin, LOW); // Turn the buzzer off
}
