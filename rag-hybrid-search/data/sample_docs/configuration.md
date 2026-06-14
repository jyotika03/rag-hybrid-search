# Configuration

Retrieval is tunable via environment variables. DENSE_WEIGHT and SPARSE_WEIGHT
control the RRF fusion balance (default 0.7 / 0.3). FINAL_TOP_K sets how many
chunks reach the generator. CONFIDENCE_THRESHOLD controls when the system
declines to answer.
