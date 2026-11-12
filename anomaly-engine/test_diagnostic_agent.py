
import logging
from diagnostic_agent import diagnose_new_anomalies

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    result = diagnose_new_anomalies()
    print("\nDiagnostic Agent Results:")
    print(result)
