# HLHFM + Eidolon-Ω Production Deployment

**Production-grade holographic memory and music generation system**

## 🚀 Overview

This repository contains a scalable, fault-tolerant implementation of the HLHFM (HyperLiquid Holographic Fractal Memory) combined with Eidolon-Ω, a transformer-based text-to-music generation system.

## 🏗️ Architecture

The system is built as a modular microservices architecture:

| Service | Purpose | Port |
|---------|---------|------|
| **hlhfm-core** | Holographic memory storage & retrieval | 8000 |
| **eidolon-inference** | Music generation (transformer + diffusion) | 8001 |
| **preprocessor** | Audio/text preprocessing | 8002 |
| **evaluator** | Real-time FAD/CLAP scoring | 8003 |
| **api-gateway** | Unified REST/gRPC interface | 8080 |
| **redis** | Persistent storage backend | 6379 |

## 🛠️ Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.9+
- Kubernetes cluster (for production deployment)

### Local Development

```bash
# Clone the repository
git clone https://github.com/MASSIVEMAGNETICS/haleon.git
cd haleon

# Start all services with Docker Compose
docker-compose up -d

# Check service health
curl http://localhost:8080/health
```

### Generate Music

```bash
# Via API Gateway
curl -X POST http://localhost:8080/generate_music \
  -H "Content-Type: application/json" \
  -d '{
    "text_prompt": "calm piano melody with nature sounds",
    "use_memory": true,
    "temperature": 1.0
  }'
```

## 📦 Service Details

### HLHFM Core

Redis-backed holographic memory with fault tolerance:

- **Endpoints:**
  - `POST /write` - Write memory entry
  - `POST /query` - Query memories
  - `GET /stats` - Get statistics
  - `POST /erase` - GDPR-compliant data erasure

- **Features:**
  - Persistent Redis storage
  - Circuit breaker pattern
  - Retry logic with exponential backoff
  - Prometheus metrics

### Eidolon-Ω Inference

ONNX-optimized music generation:

- **Endpoints:**
  - `POST /generate` - Generate music from text
  - `POST /generate_with_hlhfm` - Memory-augmented generation

- **Features:**
  - Low-latency ONNX Runtime inference
  - Async processing
  - Auto-scaling with Kubernetes HPA

### API Gateway

Unified interface orchestrating all services:

- **Endpoints:**
  - `POST /generate_music` - End-to-end generation pipeline
  - `GET /stats` - System-wide statistics
  - `GET /health` - Health check

## 🚢 Production Deployment

### Kubernetes

```bash
# Deploy Redis
kubectl apply -f k8s/redis-deployment.yaml

# Deploy HLHFM Core
kubectl apply -f k8s/hlhfm-deployment.yaml

# Deploy Eidolon Inference
kubectl apply -f k8s/eidolon-deployment.yaml

# Deploy Ingress
kubectl apply -f k8s/ingress.yaml
```

### Auto-Scaling

Horizontal Pod Autoscaler (HPA) is configured for Eidolon:

```yaml
minReplicas: 2
maxReplicas: 10
targetCPUUtilization: 70%
targetMemoryUtilization: 80%
```

## 📊 Monitoring

### Prometheus Metrics

- `hlhfm_entries_total` - Total entries in memory
- `hlhfm_query_latency_ms` - Query latency (histogram)
- `eidolon_inference_latency_ms` - Inference latency (histogram)
- `eidolon_fad_score` - Frechet Audio Distance
- `eidolon_clap_score` - CLAP similarity score

### Grafana Dashboard

Import `k8s/grafana-dashboard.json` for pre-configured visualizations.

Access metrics at: `http://localhost:8000/metrics`

## 🔒 Security Features

- **Rate Limiting:** 10 writes/min, 20 queries/min per client
- **TLS Encryption:** In-transit encryption via Ingress
- **GDPR Compliance:** User data erasure endpoint
- **Circuit Breaker:** Fault isolation with PyBreaker

## 🧪 CI/CD Pipeline

GitHub Actions workflow (`.github/workflows/deploy.yml`):

1. **Test** - Run unit tests
2. **Build** - Build Docker images
3. **Deploy** - Deploy to Kubernetes

## 📈 Performance

### Load Testing Results

Target: **10,000 RPS**

| Metric | Target | Current |
|--------|--------|---------|
| HLHFM Query Latency (p99) | <100ms | TBD |
| Eidolon Inference (p99) | <2s | TBD |
| Availability | 99.9% | TBD |

## 🔧 Configuration

Environment variables:

### HLHFM Core
```bash
REDIS_URL=redis://localhost:6379/0
HLHFM_DIM=512
HLHFM_DECAY_RATE=0.1
```

### Eidolon Inference
```bash
ONNX_MODEL_PATH=/app/eidolon.onnx
```

## 🗺️ Roadmap

- [x] Modular microservices architecture
- [x] Redis-backed HLHFM
- [x] ONNX Runtime inference
- [x] Docker Compose setup
- [x] Kubernetes manifests
- [x] CI/CD pipeline
- [x] Prometheus metrics
- [x] Rate limiting & security
- [ ] Load testing (10k RPS)
- [ ] Production deployment
- [ ] Real FAD/CLAP models
- [ ] GPU acceleration

## 📝 License

Research project - see LICENSE file

## 🤝 Contributing

This is a research project. For questions or collaboration, please open an issue.

---

**Status:** ⏳ Pre-production (Load testing in progress)

**Last Updated:** 2023-11-23
