
import time
import logging
from enhanced_client import FileClient, create_test_file, run_threading_stress_test
import pandas as pd

logging.basicConfig(level=logging.INFO)

def quick_stress_test():
    """Run quick stress test with 4 combinations"""
    results = []
    
    # Create small test file
    print("Creating test file...")
    test_file = create_test_file("quick_test.bin", 10)  # 10MB file
    with open(test_file, 'rb') as f:
        file_data = f.read()
    
    # Test configurations (4 combinations: 2 operations × 2 worker configs)
    configs = [
        {'op': 'upload', 'workers': 1, 'server_port': 45000, 'mode': 'thread'},
        {'op': 'upload', 'workers': 5, 'server_port': 45000, 'mode': 'thread'},
        {'op': 'download', 'workers': 1, 'server_port': 45000, 'mode': 'thread'},
        {'op': 'download', 'workers': 5, 'server_port': 45000, 'mode': 'thread'},
    ]
    
    print(f"Running {len(configs)} test configurations...")
    
    for i, config in enumerate(configs, 1):
        print(f"\nTest {i}/{len(configs)}: {config['op']} with {config['workers']} workers ({config['mode']} mode)")
        
        try:
            # For download tests, upload first
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
                'Nomor': i,
                'Execution_Mode': config['mode'],
                'Operasi': config['op'],
                'Volume': 10,  # 10MB test file
                'Jumlah_Client_Worker_Pool': config['workers'],
                'Jumlah_Server_Worker_Pool': 3,  # Default server workers
                'Waktu_Total_Per_Client': result['avg_time_per_client'],
                'Throughput_Per_Client': result['throughput'],
                'Client_Worker_Sukses': result['successful_workers'],
                'Client_Worker_Gagal': result['failed_workers'],
                'Throughput_MB_per_sec': result['throughput'] / (1024 * 1024)
            }
            results.append(row)
            
            print(f"  Result: {result['successful_workers']} success, {result['failed_workers']} failed")
            print(f"  Throughput: {result['throughput'] / (1024 * 1024):.2f} MB/s")
            
        except Exception as e:
            print(f"  Error: {e}")
            results.append({
                'Nomor': i,
                'Execution_Mode': config['mode'],
                'Operasi': config['op'], 
                'Volume': 10,
                'Jumlah_Client_Worker_Pool': config['workers'],
                'Jumlah_Server_Worker_Pool': 3,
                'Waktu_Total_Per_Client': 0,
                'Throughput_Per_Client': 0,
                'Client_Worker_Sukses': 0,
                'Client_Worker_Gagal': config['workers'],
                'Throughput_MB_per_sec': 0
            })
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv('quick_test_results.csv', index=False)
    
    print("\n" + "=" * 50)
    print("QUICK TEST RESULTS:")
    print("=" * 50)
    print(df.to_string(index=False))
    print(f"\nResults saved to quick_test_results.csv")
    
    return len([r for r in results if r['Client_Worker_Gagal'] == 0])

if __name__ == "__main__":
    successful_tests = quick_stress_test()
    print(f"\nQuick test completed: {successful_tests} successful configurations")
