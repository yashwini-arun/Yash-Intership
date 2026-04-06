SCENARIOS = {
    "db_overload": {
        "alert": {
            "id": "INC-2024-001",
            "service": "payment-service",
            "message": "Database CPU at 98%, query response time 12000ms (threshold: 500ms)",
            "metric": "db.cpu.percent",
            "value": 98.2,
            "threshold": 80,
            "environment": "production",
            "region": "us-east-1",
            "timestamp": "2024-01-15T02:34:11Z",
            "tags": ["database", "payment", "critical"]
        },
        "logs": """
2024-01-15T02:30:01Z [payment-service] INFO  Request received: POST /api/v1/charge
2024-01-15T02:30:02Z [payment-service] WARN  DB query slow: SELECT * FROM transactions WHERE user_id=? took 4200ms
2024-01-15T02:31:15Z [payment-service] WARN  DB connection pool exhausted (100/100 connections used)
2024-01-15T02:31:22Z [payment-service] ERROR DB query timeout after 10000ms: SELECT * FROM transactions WHERE created_at > ?
2024-01-15T02:31:45Z [postgres-primary] ERROR autovacuum: VACUUM transactions: found 2847291 dead tuples, 0 live
2024-01-15T02:32:00Z [postgres-primary] WARN  Table bloat detected on transactions table: 14GB bloat vs 2GB live data
2024-01-15T02:32:10Z [payment-service] ERROR Timeout processing payment for user_id=48291 after 12000ms
2024-01-15T02:32:30Z [payment-service] ERROR 503 Service Unavailable - circuit breaker OPEN for db-primary
2024-01-15T02:33:00Z [load-balancer]   WARN  Health check failing for payment-service (3/5 instances DOWN)
2024-01-15T02:33:45Z [payment-service] ERROR Batch job analytics_aggregator started at 02:30 consuming 60% of DB CPU
2024-01-15T02:34:11Z [datadog]         ALERT DB CPU threshold breached: 98.2% > 80% for 4 minutes
"""
    },
    "memory_leak": {
        "alert": {
            "id": "INC-2024-002",
            "service": "api-gateway",
            "message": "api-gateway pod memory at 94% (7.5GB/8GB), OOMKill imminent",
            "metric": "container.memory.usage_percent",
            "value": 94.1,
            "threshold": 85,
            "environment": "production",
            "region": "us-west-2",
            "timestamp": "2024-01-16T14:17:33Z",
            "tags": ["kubernetes", "memory", "api-gateway"]
        },
        "logs": """
2024-01-16T12:00:00Z [api-gateway] INFO  Pod started, memory: 512MB
2024-01-16T12:30:00Z [api-gateway] INFO  Memory usage: 1.1GB (normal growth)
2024-01-16T13:00:00Z [api-gateway] INFO  Memory usage: 2.8GB (elevated)
2024-01-16T13:30:00Z [api-gateway] WARN  Memory usage: 4.9GB (high) - GC not reclaiming
2024-01-16T13:45:00Z [api-gateway] WARN  Heap dump shows 2.1M uncollected RequestContext objects
2024-01-16T14:00:00Z [api-gateway] WARN  Memory usage: 6.2GB - response latency increasing
2024-01-16T14:05:00Z [api-gateway] ERROR GC overhead limit exceeded - 98% time in garbage collection
2024-01-16T14:10:00Z [api-gateway] ERROR Memory usage: 7.1GB - dropping 12% of requests
2024-01-16T14:15:00Z [api-gateway] ERROR Memory usage: 7.5GB - OOMKill risk HIGH
2024-01-16T14:15:30Z [k8s-system]  WARN  Pod api-gateway-7d9f8b-xk2p9 approaching memory limit (8GB)
2024-01-16T14:16:00Z [api-gateway] WARN  Deploy v2.4.1 shipped at 11:58 included new middleware that doesn't close connections
2024-01-16T14:17:33Z [prometheus]  ALERT container.memory.usage_percent=94.1 FIRING
"""
    },
    "bad_deployment": {
        "alert": {
            "id": "INC-2024-003",
            "service": "user-service",
            "message": "Error rate spiked to 34% (threshold: 1%) after deployment v3.2.0",
            "metric": "http.error_rate_percent",
            "value": 34.2,
            "threshold": 1.0,
            "environment": "production",
            "region": "eu-west-1",
            "timestamp": "2024-01-17T09:45:22Z",
            "tags": ["deployment", "error-rate", "user-service"]
        },
        "logs": """
2024-01-17T09:30:00Z [deploy-system] INFO  Starting deployment user-service v3.2.0 (rolling update)
2024-01-17T09:31:00Z [deploy-system] INFO  Replacing pod user-service-v3.1.9-abc123 with v3.2.0-def456
2024-01-17T09:32:00Z [user-service]  INFO  v3.2.0 pod started and healthy
2024-01-17T09:33:00Z [user-service]  ERROR NullPointerException at UserProfileMapper.java:247
2024-01-17T09:33:01Z [user-service]  ERROR NullPointerException at UserProfileMapper.java:247
2024-01-17T09:33:10Z [user-service]  ERROR 500 Internal Server Error: GET /api/v1/users/profile - 847ms
2024-01-17T09:35:00Z [deploy-system] INFO  50% of traffic now on v3.2.0 (rolling continues)
2024-01-17T09:38:00Z [user-service]  ERROR Error rate: 22% and rising
2024-01-17T09:40:00Z [user-service]  ERROR NullPointerException: new optional field 'preferences' not handled in mapper
2024-01-17T09:42:00Z [deploy-system] INFO  Deployment complete: 100% traffic on v3.2.0
2024-01-17T09:44:00Z [user-service]  ERROR Error rate: 34.2% - all affected users missing 'preferences' field (DB migration pending)
2024-01-17T09:45:22Z [datadog]       ALERT http.error_rate_percent=34.2 FIRING for user-service
"""
    }
}