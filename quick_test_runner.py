
import time
import logging
from enhanced_client import FileClient, create_test_file, run_threading_stress_test
import pandas as pd

logging.basicConfig(level=logging.INFO)

def quick_stress_test():
    """Run quick stress test"""
    results = []
    
    # Create small test file
    print("Creating test file...")
    test_file = create_test_file("quick_test.bin", 1)  # 1MB file
    with open(test_file, 'rb') as f:
        file_data = f.read()
    
    # Test configurations (simplified)
    configs = [
        {'op': 'upload', 'workers': 1, 'server_port': 45000},
        {'op': 'upload', 'workers': 5, 'server_port': 45000},
        {'op': 'download', 'workers': 1, 'server_port': 45000},
        {'op': 'download', 'workers': 5, 'server_port': 45000},
    ]
    
    print(f"Running {len(configs)} test configurations...")
    
    for i, config in enumerate(configs, 1):
        print(f"\nTest {i}/{len(configs)}: {config['op']} with {config['workers']} workers")
        
        try:
            # For download tests, we'd need to upload first
            if config['op'] == 'download':
                client = FileClient('localhost', config['server_port'])
                client.upload_file("quick_test.bin", file_data)
            
            result = run_threading_stress_test(
                config['op'],
                'localhost', 
                config['server_port'],
                "quick_test.bin",
                file_data,
                config['workers'],
                timeout=30
            )
            
            row = {
                'Test': i,
                'Operation': config['op'],
                'Workers': config['workers'],
                'Success': result['successful_workers'],
                'Failed': result['failed_workers'],
                'Avg_Time': result['avg_time_per_client'],
                'Throughput': result['throughput']
            }
            results.append(row)
            
            print(f"  Result: {result['successful_workers']} success, {result['failed_workers']} failed")
            print(f"  Throughput: {result['throughput']:.2f} B/s")
            
        except Exception as e:
            print(f"  Error: {e}")
            results.append({
                'Test': i,
                'Operation': config['op'], 
                'Workers': config['workers'],
                'Success': 0,
                'Failed': config['workers'],
                'Avg_Time': 0,
                'Throughput': 0
            })
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv('quick_test_results.csv', index=False)
    
    print("\n" + "=" * 40)
    print("QUICK TEST RESULTS:")
    print("=" * 40)
    print(df.to_string(index=False))
    print(f"\nResults saved to quick_test_results.csv")
    
    return len([r for r in results if r['Failed'] == 0])

if __name__ == "__main__":
    successful_tests = quick_stress_test()
    print(f"\nQuick test completed: {successful_tests} successful configurations")
