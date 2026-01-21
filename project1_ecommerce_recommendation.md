# Mini-Project DevOps/MLOps: E-commerce Recommendation System

## Objective
Build and deploy a product recommendation system demonstrating complete DevOps/MLOps practices.

## Project Description
Create an API that recommends products to users based on their history. The system must include:
- An ML recommendation model (collaborative filtering or content-based)
- A REST API to serve predictions
- A simple web interface for demonstration

**Data**: Amazon Product Reviews dataset or synthetic data

---

## Technical Requirements

### 1. Version Control
- Git repository with branching strategy (main, develop, feature branches)
- Pull requests with code reviews before merging

### 2. CI/CD Pipeline
- Automated testing on every push
- Automated Docker image builds
- Automated deployment to staging and production
- Rollback capability when deployment fails

### 3. Containerization
- Create Dockerfiles for all services
- Push images to a container registry

### 4. Container Orchestration
- Deploy application on Kubernetes
- Configure multiple replicas, services, health checks
- Implement auto-scaling based on load

### 5. Model Management (MLflow)
- Track experiments with different hyperparameters
- Register models with versions in model registry
- Store model metadata (metrics, parameters, artifacts)

### 6. Monitoring
- Collect and visualize metrics (system, API, model performance)
- Create dashboards showing health and performance
- Set up alerts for critical issues

### 7. Testing
- Write unit tests
- Write integration tests for API
- Write model validation tests
- Perform load testing

### 8. Deployment Strategy
- Implement one strategy: Blue-Green, Canary, or Rolling Update
- Demonstrate the deployment process

### 9. Model Retraining Pipeline
- Create automated pipeline to retrain model
- Validate data before training
- Evaluate new model performance
- Promote to production only if better than current model

---

## Deliverable

**Final Presentation (30 minutes)**

Your presentation must include:

1. **Explanation of Implementation** (10 minutes)
   - Architecture overview
   - Technical choices and justifications
   - Challenges faced and solutions

2. **Live Demonstration** (15 minutes)
   - Application functionality (make predictions via API/web interface)
   - CI/CD pipeline execution (trigger build, show automated deployment)
   - Monitoring dashboards (system metrics, API performance, model metrics)
   - MLflow tracking (experiments, model registry)
   - Deployment strategy in action
   - Model retraining pipeline execution

3. **Q&A** (5 minutes)

All team members must participate in the presentation.
