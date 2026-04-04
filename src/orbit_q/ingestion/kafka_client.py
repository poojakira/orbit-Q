import json
import logging
from typing import Any, Dict, List, Optional
from confluent_kafka import Producer, Consumer, KafkaError
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OrbitKafkaClient:
    """
    High-throughput Kafka client for Orbit-Q telemetry.
    Supports both production (Simulator) and consumption (Orchestrator).
    """

    def __init__(self) -> None:
        self.bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
        self.topic = os.getenv("KAFKA_TELEMETRY_TOPIC", "orbit_telemetry")
        
        # Producer Configuration
        self.producer_conf = {
            'bootstrap.servers': self.bootstrap_servers,
            'client.id': 'orbit_producer',
            'compression.type': 'snappy',
            'linger.ms': 20,
            'acks': 'all'
        }
        
        # Consumer Configuration
        self.consumer_conf = {
            'bootstrap.servers': self.bootstrap_servers,
            'group.id': 'orbit_orchestrator_group',
            'auto.offset.reset': 'latest',
            'enable.auto.commit': True
        }
        
        self.producer: Optional[Producer] = None
        self.consumer: Optional[Consumer] = None

    def _init_producer(self) -> None:
        if not self.producer:
            try:
                self.producer = Producer(self.producer_conf)
                logger.info(f"Kafka Producer initialized on {self.bootstrap_servers}")
            except Exception as e:
                logger.error(f"Failed to initialize Kafka Producer: {e}")

    def _init_consumer(self) -> None:
        if not self.consumer:
            try:
                self.consumer = Consumer(self.consumer_conf)
                self.consumer.subscribe([self.topic])
                logger.info(f"Kafka Consumer initialized and subscribed to {self.topic}")
            except Exception as e:
                logger.error(f"Failed to initialize Kafka Consumer: {e}")

    def produce_telemetry(self, data: Dict[str, Any]) -> None:
        """Sends a telemetry packet to the Kafka topic."""
        self._init_producer()
        if self.producer:
            try:
                self.producer.produce(
                    self.topic, 
                    value=json.dumps(data).encode('utf-8'),
                    callback=self._delivery_report
                )
                self.producer.poll(0)
            except BufferError:
                logger.warning("Kafka Producer buffer full, retrying...")
                self.producer.poll(1)
                self.produce_telemetry(data)

    def consume_batch(self, limit: int = 500, timeout: float = 1.0) -> List[Dict[str, Any]]:
        """Consumes a batch of telemetry packets from Kafka."""
        self._init_consumer()
        batch = []
        if not self.consumer:
            return batch

        msgs = self.consumer.consume(num_messages=limit, timeout=timeout)
        for msg in msgs:
            if msg.error():
                if msg.error().code() != KafkaError._PARTITION_EOF:
                    logger.error(f"Kafka Consumer error: {msg.error()}")
                continue
            
            try:
                batch.append(json.loads(msg.value().decode('utf-8')))
            except Exception as e:
                logger.error(f"Failed to parse Kafka message: {e}")
        
        return batch

    def _delivery_report(self, err: Any, msg: Any) -> None:
        if err is not None:
            logger.error(f"Message delivery failed: {err}")

    def flush(self) -> None:
        if self.producer:
            self.producer.flush()

    def close(self) -> None:
        if self.consumer:
            self.consumer.close()
