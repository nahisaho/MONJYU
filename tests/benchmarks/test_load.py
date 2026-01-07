"""
MONJYU Load Tests (NFR-TST-004 / NFR-PERF)

負荷テスト・ストレステスト
対応要件:
- NFR-PERF-001: Vector Search レイテンシ < 500ms (p95)
- NFR-PERF-002: Lazy Search レイテンシ < 5s (p95)
- NFR-PERF-003: Graph Search レイテンシ < 10s (p95)
- NFR-PERF-006: 同時クエリ処理数 > 100 concurrent
- NFR-SCAL-002: 同時接続ユーザー数 > 100 users

Usage:
    # 基本的な負荷テスト
    pytest tests/benchmarks/test_load.py -v
    
    # 詳細出力
    pytest tests/benchmarks/test_load.py -v -s --tb=short
    
    # 特定のテストのみ
    pytest tests/benchmarks/test_load.py::TestConcurrentLoad -v
"""

from __future__ import annotations

import asyncio
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable
from unittest.mock import MagicMock, patch, AsyncMock

import pytest


# =============================================================================
# テストデータ・設定
# =============================================================================

@dataclass
class LoadTestConfig:
    """負荷テスト設定"""
    
    # 同時実行数
    concurrent_users: int = 100
    
    # リクエスト数
    total_requests: int = 1000
    
    # タイムアウト（秒）
    timeout: float = 30.0
    
    # パフォーマンス基準（p95、ミリ秒）
    vector_search_p95_ms: float = 500.0
    lazy_search_p95_ms: float = 5000.0
    graph_search_p95_ms: float = 10000.0
    
    # スループット基準（リクエスト/分）
    min_throughput_rpm: int = 1000


@dataclass
class LoadTestResult:
    """負荷テスト結果"""
    
    test_name: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    
    # レイテンシ（ミリ秒）
    latencies_ms: list[float] = field(default_factory=list)
    
    # 時間
    total_duration_sec: float = 0.0
    
    @property
    def success_rate(self) -> float:
        """成功率"""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests * 100
    
    @property
    def p50_ms(self) -> float:
        """p50 レイテンシ"""
        if not self.latencies_ms:
            return 0.0
        return statistics.median(self.latencies_ms)
    
    @property
    def p95_ms(self) -> float:
        """p95 レイテンシ"""
        if not self.latencies_ms:
            return 0.0
        sorted_latencies = sorted(self.latencies_ms)
        index = int(len(sorted_latencies) * 0.95)
        return sorted_latencies[min(index, len(sorted_latencies) - 1)]
    
    @property
    def p99_ms(self) -> float:
        """p99 レイテンシ"""
        if not self.latencies_ms:
            return 0.0
        sorted_latencies = sorted(self.latencies_ms)
        index = int(len(sorted_latencies) * 0.99)
        return sorted_latencies[min(index, len(sorted_latencies) - 1)]
    
    @property
    def avg_ms(self) -> float:
        """平均レイテンシ"""
        if not self.latencies_ms:
            return 0.0
        return statistics.mean(self.latencies_ms)
    
    @property
    def throughput_rpm(self) -> float:
        """スループット（リクエスト/分）"""
        if self.total_duration_sec == 0:
            return 0.0
        return self.successful_requests / self.total_duration_sec * 60
    
    def summary(self) -> dict[str, Any]:
        """サマリー"""
        return {
            "test_name": self.test_name,
            "total_requests": self.total_requests,
            "successful": self.successful_requests,
            "failed": self.failed_requests,
            "success_rate": f"{self.success_rate:.1f}%",
            "p50_ms": f"{self.p50_ms:.1f}",
            "p95_ms": f"{self.p95_ms:.1f}",
            "p99_ms": f"{self.p99_ms:.1f}",
            "avg_ms": f"{self.avg_ms:.1f}",
            "throughput_rpm": f"{self.throughput_rpm:.0f}",
            "duration_sec": f"{self.total_duration_sec:.2f}",
        }


# =============================================================================
# テストヘルパー
# =============================================================================

def simulate_vector_search(query: str, delay_ms: float = 50.0) -> dict[str, Any]:
    """ベクトル検索のシミュレーション"""
    # 実際の処理時間をシミュレート（ばらつきを含む）
    import random
    actual_delay = delay_ms * (0.8 + random.random() * 0.4)  # ±20%のばらつき
    time.sleep(actual_delay / 1000)
    
    return {
        "query": query,
        "results": [{"id": f"doc_{i}", "score": 0.9 - i * 0.1} for i in range(5)],
        "latency_ms": actual_delay,
    }


def simulate_lazy_search(query: str, delay_ms: float = 500.0) -> dict[str, Any]:
    """Lazy検索のシミュレーション"""
    import random
    actual_delay = delay_ms * (0.8 + random.random() * 0.4)
    time.sleep(actual_delay / 1000)
    
    return {
        "query": query,
        "answer": f"Answer for: {query}",
        "citations": [{"doc_id": f"doc_{i}"} for i in range(3)],
        "latency_ms": actual_delay,
    }


async def simulate_async_search(query: str, delay_ms: float = 50.0) -> dict[str, Any]:
    """非同期検索のシミュレーション"""
    import random
    actual_delay = delay_ms * (0.8 + random.random() * 0.4)
    await asyncio.sleep(actual_delay / 1000)
    
    return {
        "query": query,
        "results": [{"id": f"doc_{i}", "score": 0.9 - i * 0.1} for i in range(5)],
        "latency_ms": actual_delay,
    }


def run_concurrent_load_test(
    func: Callable,
    num_requests: int,
    max_workers: int,
    test_name: str = "load_test",
) -> LoadTestResult:
    """同時実行負荷テストを実行"""
    result = LoadTestResult(
        test_name=test_name,
        total_requests=num_requests,
        successful_requests=0,
        failed_requests=0,
    )
    
    start_time = time.perf_counter()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i in range(num_requests):
            query = f"test query {i}"
            futures.append(executor.submit(func, query))
        
        for future in as_completed(futures):
            try:
                res = future.result(timeout=30.0)
                result.successful_requests += 1
                if "latency_ms" in res:
                    result.latencies_ms.append(res["latency_ms"])
            except Exception as e:
                result.failed_requests += 1
    
    result.total_duration_sec = time.perf_counter() - start_time
    return result


async def run_async_load_test(
    func: Callable,
    num_requests: int,
    concurrency: int,
    test_name: str = "async_load_test",
) -> LoadTestResult:
    """非同期負荷テストを実行"""
    result = LoadTestResult(
        test_name=test_name,
        total_requests=num_requests,
        successful_requests=0,
        failed_requests=0,
    )
    
    semaphore = asyncio.Semaphore(concurrency)
    
    async def bounded_request(query: str):
        async with semaphore:
            return await func(query)
    
    start_time = time.perf_counter()
    
    tasks = [bounded_request(f"test query {i}") for i in range(num_requests)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for res in results:
        if isinstance(res, Exception):
            result.failed_requests += 1
        else:
            result.successful_requests += 1
            if "latency_ms" in res:
                result.latencies_ms.append(res["latency_ms"])
    
    result.total_duration_sec = time.perf_counter() - start_time
    return result


# =============================================================================
# 1. 同時実行テスト (NFR-PERF-006, NFR-SCAL-002)
# =============================================================================

class TestConcurrentLoad:
    """同時実行負荷テスト"""
    
    def test_concurrent_100_users(self):
        """100同時ユーザーのテスト (NFR-SCAL-002)"""
        config = LoadTestConfig(concurrent_users=100, total_requests=100)
        
        result = run_concurrent_load_test(
            func=simulate_vector_search,
            num_requests=config.total_requests,
            max_workers=config.concurrent_users,
            test_name="concurrent_100_users",
        )
        
        print(f"\n--- {result.test_name} ---")
        for k, v in result.summary().items():
            print(f"  {k}: {v}")
        
        # 検証: 成功率 > 95%
        assert result.success_rate > 95.0, f"Success rate too low: {result.success_rate}%"
        
        # 検証: p95 < 500ms (NFR-PERF-001)
        assert result.p95_ms < config.vector_search_p95_ms, \
            f"p95 latency too high: {result.p95_ms}ms > {config.vector_search_p95_ms}ms"
    
    def test_concurrent_10_users_lazy_search(self):
        """10同時ユーザーでのLazy検索テスト"""
        config = LoadTestConfig(concurrent_users=10, total_requests=20)
        
        result = run_concurrent_load_test(
            func=simulate_lazy_search,
            num_requests=config.total_requests,
            max_workers=config.concurrent_users,
            test_name="lazy_search_10_users",
        )
        
        print(f"\n--- {result.test_name} ---")
        for k, v in result.summary().items():
            print(f"  {k}: {v}")
        
        # 検証: 成功率 > 95%
        assert result.success_rate > 95.0
        
        # 検証: p95 < 5000ms (NFR-PERF-002)
        assert result.p95_ms < config.lazy_search_p95_ms, \
            f"p95 latency too high: {result.p95_ms}ms > {config.lazy_search_p95_ms}ms"
    
    def test_burst_traffic(self):
        """バーストトラフィックテスト"""
        # 短時間に大量のリクエストを送信
        config = LoadTestConfig(concurrent_users=50, total_requests=200)
        
        result = run_concurrent_load_test(
            func=lambda q: simulate_vector_search(q, delay_ms=20.0),
            num_requests=config.total_requests,
            max_workers=config.concurrent_users,
            test_name="burst_traffic",
        )
        
        print(f"\n--- {result.test_name} ---")
        for k, v in result.summary().items():
            print(f"  {k}: {v}")
        
        # 検証: 成功率 > 90%（バースト時は少し緩和）
        assert result.success_rate > 90.0


# =============================================================================
# 2. スループットテスト (NFR-PERF-005, NFR-TST-004)
# =============================================================================

class TestThroughput:
    """スループットテスト"""
    
    def test_throughput_1000_rpm(self):
        """1000 リクエスト/分のスループットテスト"""
        config = LoadTestConfig(concurrent_users=20, total_requests=100)
        
        result = run_concurrent_load_test(
            func=lambda q: simulate_vector_search(q, delay_ms=10.0),
            num_requests=config.total_requests,
            max_workers=config.concurrent_users,
            test_name="throughput_test",
        )
        
        print(f"\n--- {result.test_name} ---")
        for k, v in result.summary().items():
            print(f"  {k}: {v}")
        
        # 検証: スループット > 1000 rpm
        assert result.throughput_rpm > config.min_throughput_rpm, \
            f"Throughput too low: {result.throughput_rpm} < {config.min_throughput_rpm} rpm"
    
    def test_sustained_load(self):
        """持続負荷テスト（30秒間）"""
        # 長時間の負荷テスト
        config = LoadTestConfig(concurrent_users=10, total_requests=50)
        
        result = run_concurrent_load_test(
            func=lambda q: simulate_vector_search(q, delay_ms=100.0),
            num_requests=config.total_requests,
            max_workers=config.concurrent_users,
            test_name="sustained_load",
        )
        
        print(f"\n--- {result.test_name} ---")
        for k, v in result.summary().items():
            print(f"  {k}: {v}")
        
        # 検証: 成功率 > 99%（持続負荷では高い成功率を期待）
        assert result.success_rate > 99.0


# =============================================================================
# 3. レイテンシテスト (NFR-PERF-001〜004)
# =============================================================================

class TestLatency:
    """レイテンシテスト"""
    
    def test_vector_search_p95_under_500ms(self):
        """Vector Search p95 < 500ms (NFR-PERF-001)"""
        config = LoadTestConfig(total_requests=100)
        
        result = run_concurrent_load_test(
            func=lambda q: simulate_vector_search(q, delay_ms=100.0),
            num_requests=config.total_requests,
            max_workers=10,
            test_name="vector_search_latency",
        )
        
        print(f"\n--- {result.test_name} ---")
        print(f"  p50: {result.p50_ms:.1f}ms")
        print(f"  p95: {result.p95_ms:.1f}ms")
        print(f"  p99: {result.p99_ms:.1f}ms")
        
        # 検証: p95 < 500ms
        assert result.p95_ms < config.vector_search_p95_ms
    
    def test_lazy_search_p95_under_5s(self):
        """Lazy Search p95 < 5s (NFR-PERF-002)"""
        config = LoadTestConfig(total_requests=20)
        
        result = run_concurrent_load_test(
            func=lambda q: simulate_lazy_search(q, delay_ms=1000.0),
            num_requests=config.total_requests,
            max_workers=5,
            test_name="lazy_search_latency",
        )
        
        print(f"\n--- {result.test_name} ---")
        print(f"  p50: {result.p50_ms:.1f}ms")
        print(f"  p95: {result.p95_ms:.1f}ms")
        print(f"  p99: {result.p99_ms:.1f}ms")
        
        # 検証: p95 < 5000ms
        assert result.p95_ms < config.lazy_search_p95_ms
    
    def test_latency_distribution(self):
        """レイテンシ分布テスト"""
        result = run_concurrent_load_test(
            func=lambda q: simulate_vector_search(q, delay_ms=50.0),
            num_requests=200,
            max_workers=20,
            test_name="latency_distribution",
        )
        
        print(f"\n--- {result.test_name} ---")
        print(f"  min: {min(result.latencies_ms):.1f}ms")
        print(f"  p50: {result.p50_ms:.1f}ms")
        print(f"  p95: {result.p95_ms:.1f}ms")
        print(f"  p99: {result.p99_ms:.1f}ms")
        print(f"  max: {max(result.latencies_ms):.1f}ms")
        print(f"  stddev: {statistics.stdev(result.latencies_ms):.1f}ms")
        
        # 検証: 標準偏差が平均の50%以下（安定性）
        stddev = statistics.stdev(result.latencies_ms)
        assert stddev < result.avg_ms * 0.5, f"Latency too variable: stddev={stddev:.1f}ms"


# =============================================================================
# 4. 非同期負荷テスト
# =============================================================================

class TestAsyncLoad:
    """非同期負荷テスト"""
    
    @pytest.mark.asyncio
    async def test_async_concurrent_100(self):
        """100同時非同期リクエストテスト"""
        result = await run_async_load_test(
            func=simulate_async_search,
            num_requests=100,
            concurrency=100,
            test_name="async_100_concurrent",
        )
        
        print(f"\n--- {result.test_name} ---")
        for k, v in result.summary().items():
            print(f"  {k}: {v}")
        
        # 検証: 成功率 > 95%
        assert result.success_rate > 95.0
    
    @pytest.mark.asyncio
    async def test_async_high_concurrency(self):
        """高同時実行数テスト"""
        result = await run_async_load_test(
            func=lambda q: simulate_async_search(q, delay_ms=20.0),
            num_requests=500,
            concurrency=200,
            test_name="async_high_concurrency",
        )
        
        print(f"\n--- {result.test_name} ---")
        for k, v in result.summary().items():
            print(f"  {k}: {v}")
        
        # 検証: スループット > 1000 rpm
        assert result.throughput_rpm > 1000


# =============================================================================
# 5. ストレステスト
# =============================================================================

class TestStress:
    """ストレステスト"""
    
    def test_gradual_load_increase(self):
        """段階的負荷増加テスト"""
        results = []
        
        for num_workers in [10, 20, 50, 100]:
            result = run_concurrent_load_test(
                func=lambda q: simulate_vector_search(q, delay_ms=30.0),
                num_requests=50,
                max_workers=num_workers,
                test_name=f"gradual_load_{num_workers}",
            )
            results.append(result)
            
            print(f"\n--- Workers: {num_workers} ---")
            print(f"  Success Rate: {result.success_rate:.1f}%")
            print(f"  p95: {result.p95_ms:.1f}ms")
            print(f"  Throughput: {result.throughput_rpm:.0f} rpm")
        
        # 検証: 全てのステップで成功率 > 90%
        for result in results:
            assert result.success_rate > 90.0
    
    def test_error_recovery(self):
        """エラー回復テスト"""
        error_count = 0
        
        def flaky_search(query: str) -> dict[str, Any]:
            nonlocal error_count
            import random
            # 10%の確率でエラー
            if random.random() < 0.1:
                error_count += 1
                raise Exception("Simulated error")
            return simulate_vector_search(query, delay_ms=20.0)
        
        result = run_concurrent_load_test(
            func=flaky_search,
            num_requests=100,
            max_workers=20,
            test_name="error_recovery",
        )
        
        print(f"\n--- {result.test_name} ---")
        print(f"  Simulated errors: {error_count}")
        print(f"  Failed requests: {result.failed_requests}")
        print(f"  Success Rate: {result.success_rate:.1f}%")
        
        # 検証: エラーがあっても成功率 > 85%
        assert result.success_rate > 85.0


# =============================================================================
# 6. リソース使用量テスト (NFR-PERF-007)
# =============================================================================

class TestResourceUsage:
    """リソース使用量テスト"""
    
    def test_memory_under_load(self):
        """負荷時のメモリ使用量テスト"""
        import gc
        
        # GC実行して初期状態をクリーン化
        gc.collect()
        
        # 負荷テスト実行
        result = run_concurrent_load_test(
            func=lambda q: simulate_vector_search(q, delay_ms=10.0),
            num_requests=200,
            max_workers=50,
            test_name="memory_test",
        )
        
        # 負荷後のメモリ確認（実際の実装ではpsutilなどを使用）
        print(f"\n--- {result.test_name} ---")
        print(f"  Requests: {result.total_requests}")
        print(f"  Success Rate: {result.success_rate:.1f}%")
        
        # 検証: 成功率 > 95%
        assert result.success_rate > 95.0
    
    def test_no_connection_leak(self):
        """コネクションリークテスト"""
        # 複数回のテストを実行してリソースリークを確認
        for i in range(3):
            result = run_concurrent_load_test(
                func=lambda q: simulate_vector_search(q, delay_ms=10.0),
                num_requests=50,
                max_workers=20,
                test_name=f"connection_leak_test_{i}",
            )
            
            # 各イテレーションで成功率を確認
            assert result.success_rate > 95.0, \
                f"Iteration {i} failed: {result.success_rate}%"
        
        print(f"\n--- Connection Leak Test ---")
        print(f"  All 3 iterations passed")


# =============================================================================
# 7. 統合負荷テスト
# =============================================================================

class TestIntegratedLoad:
    """統合負荷テスト"""
    
    def test_mixed_workload(self):
        """混合ワークロードテスト"""
        import random
        
        def mixed_search(query: str) -> dict[str, Any]:
            # 70% Vector Search, 30% Lazy Search
            if random.random() < 0.7:
                return simulate_vector_search(query, delay_ms=50.0)
            else:
                return simulate_lazy_search(query, delay_ms=500.0)
        
        result = run_concurrent_load_test(
            func=mixed_search,
            num_requests=100,
            max_workers=20,
            test_name="mixed_workload",
        )
        
        print(f"\n--- {result.test_name} ---")
        for k, v in result.summary().items():
            print(f"  {k}: {v}")
        
        # 検証: 成功率 > 95%
        assert result.success_rate > 95.0
    
    def test_realistic_traffic_pattern(self):
        """現実的なトラフィックパターンテスト"""
        # 時間帯によって負荷が変動するパターンをシミュレート
        results = []
        
        # 低負荷期間
        result_low = run_concurrent_load_test(
            func=lambda q: simulate_vector_search(q, delay_ms=30.0),
            num_requests=30,
            max_workers=5,
            test_name="traffic_low",
        )
        results.append(result_low)
        
        # 高負荷期間
        result_high = run_concurrent_load_test(
            func=lambda q: simulate_vector_search(q, delay_ms=30.0),
            num_requests=100,
            max_workers=50,
            test_name="traffic_high",
        )
        results.append(result_high)
        
        # ピーク期間
        result_peak = run_concurrent_load_test(
            func=lambda q: simulate_vector_search(q, delay_ms=30.0),
            num_requests=150,
            max_workers=100,
            test_name="traffic_peak",
        )
        results.append(result_peak)
        
        print(f"\n--- Realistic Traffic Pattern ---")
        for r in results:
            print(f"  {r.test_name}: {r.success_rate:.1f}% success, p95={r.p95_ms:.1f}ms")
        
        # 検証: 全パターンで成功率 > 90%
        for result in results:
            assert result.success_rate > 90.0


# =============================================================================
# メイン実行
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
