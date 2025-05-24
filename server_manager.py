import subprocess
import time
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List

class ServerManager:
    """Manage multiple server instances for testing"""
    
    def __init__(self):
        self.servers = {}
    
    def start_server(self, server_id: str, server_type: str, port: int, workers: int):
        """Start a server instance"""
        # Create a custom server script for each instance
        server_script = f"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from file_server_pools import ThreadPoolServer, ProcessPoolServer
import logging

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    if "{server_type}" == "thread":
        server = ThreadPoolServer(server_port={port}, max_workers={workers})
    else:
        server = ProcessPoolServer(server_port={port}, max_workers={workers})
    
    try:
        server.start()
    except KeyboardInterrupt:
        server.stop()
"""
        
        script_filename = f"temp_server_{server_id}.py"
        with open(script_filename, 'w') as f:
            f.write(server_script)
        
        # Start the server
        process = subprocess.Popen([sys.executable, script_filename])
        time.sleep(2)  # Wait for server to start
        
        self.servers[server_id] = {
            'process': process,
            'script_file': script_filename,
            'port': port,
            'type': server_type,
            'workers': workers
        }
        
        return process
    
    def stop_server(self, server_id: str):
        """Stop a server instance"""
        if server_id in self.servers:
            server_info = self.servers[server_id]
            process = server_info['process']
            
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            
            # Clean up script file
            try:
                os.remove(server_info['script_file'])
            except:
                pass
            
            del self.servers[server_id]
    
    def stop_all_servers(self):
        """Stop all server instances"""
        server_ids = list(self.servers.keys())
        for server_id in server_ids:
            self.stop_server(server_id)

class ResultsAnalyzer:
    """Analyze stress test results"""
    
    def __init__(self, csv_file: str):
        self.df = pd.read_csv(csv_file)
        self.setup_plotting()
    
    def setup_plotting(self):
        """Setup plotting parameters"""
        plt.style.use('default')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = (12, 8)
    
    def analyze_throughput_by_operation(self):
        """Analyze throughput by operation type"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Throughput by operation
        sns.boxplot(data=self.df, x='Operation', y='Throughput_Bytes_Per_Second', ax=axes[0,0])
        axes[0,0].set_title('Throughput by Operation')
        axes[0,0].set_ylabel('Throughput (Bytes/sec)')
        
        # Throughput by file size
        sns.boxplot(data=self.df, x='Volume_MB', y='Throughput_Bytes_Per_Second', ax=axes[0,1])
        axes[0,1].set_title('Throughput by File Size')
        axes[0,1].set_ylabel('Throughput (Bytes/sec)')
        
        # Throughput by client workers
        sns.lineplot(data=self.df, x='Client_Workers', y='Throughput_Bytes_Per_Second', 
                    hue='Operation', ax=axes[1,0])
        axes[1,0].set_title('Throughput vs Client Workers')
        axes[1,0].set_ylabel('Throughput (Bytes/sec)')
        
        # Throughput by server workers
        sns.lineplot(data=self.df, x='Server_Workers', y='Throughput_Bytes_Per_Second', 
                    hue='Operation', ax=axes[1,1])
        axes[1,1].set_title('Throughput vs Server Workers')
        axes[1,1].set_ylabel('Throughput (Bytes/sec)')
        
        plt.tight_layout()
        plt.savefig('throughput_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_success_rates(self):
        """Analyze success rates"""
        # Calculate success rates
        self.df['Client_Success_Rate'] = self.df['Client_Successful'] / \
                                        (self.df['Client_Successful'] + self.df['Client_Failed'])
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Success rate by operation
        success_by_op = self.df.groupby('Operation')['Client_Success_Rate'].mean()
        success_by_op.plot(kind='bar', ax=axes[0,0])
        axes[0,0].set_title('Success Rate by Operation')
        axes[0,0].set_ylabel('Success Rate')
        axes[0,0].set_xticklabels(axes[0,0].get_xticklabels(), rotation=0)
        
        # Success rate by file size
        success_by_size = self.df.groupby('Volume_MB')['Client_Success_Rate'].mean()
        success_by_size.plot(kind='bar', ax=axes[0,1])
        axes[0,1].set_title('Success Rate by File Size')
        axes[0,1].set_ylabel('Success Rate')
        
        # Success rate by client workers
        sns.scatterplot(data=self.df, x='Client_Workers', y='Client_Success_Rate', 
                       hue='Operation', ax=axes[1,0])
        axes[1,0].set_title('Success Rate vs Client Workers')
        
        # Success rate by server workers
        sns.scatterplot(data=self.df, x='Server_Workers', y='Client_Success_Rate', 
                       hue='Operation', ax=axes[1,1])
        axes[1,1].set_title('Success Rate vs Server Workers')
        
        plt.tight_layout()
        plt.savefig('success_rate_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_response_times(self):
        """Analyze response times"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Response time distribution
        sns.histplot(data=self.df, x='Total_Time_Seconds', hue='Operation', ax=axes[0,0])
        axes[0,0].set_title('Response Time Distribution')
        
        # Response time vs file size
        sns.boxplot(data=self.df, x='Volume_MB', y='Total_Time_Seconds', hue='Operation', ax=axes[0,1])
        axes[0,1].set_title('Response Time by File Size')
        
        # Response time vs client workers
        sns.scatterplot(data=self.df, x='Client_Workers', y='Total_Time_Seconds', 
                       hue='Operation', size='Volume_MB', ax=axes[1,0])
        axes[1,0].set_title('Response Time vs Client Workers')
        
        # Response time vs server workers
        sns.scatterplot(data=self.df, x='Server_Workers', y='Total_Time_Seconds', 
                       hue='Operation', size='Volume_MB', ax=axes[1,1])
        axes[1,1].set_title('Response Time vs Server Workers')
        
        plt.tight_layout()
        plt.savefig('response_time_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_performance_heatmap(self):
        """Generate performance heatmap"""
        # Create pivot table for heatmap
        pivot_download = self.df[self.df['Operation'] == 'download'].pivot_table(
            values='Throughput_Bytes_Per_Second',
            index='Client_Workers',
            columns='Server_Workers',
            aggfunc='mean'
        )
        
        pivot_upload = self.df[self.df['Operation'] == 'upload'].pivot_table(
            values='Throughput_Bytes_Per_Second',
            index='Client_Workers', 
            columns='Server_Workers',
            aggfunc='mean'
        )
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        sns.heatmap(pivot_download, annot=True, fmt='.0f', cmap='YlOrRd', ax=axes[0])
        axes[0].set_title('Download Throughput Heatmap\n(Client Workers vs Server Workers)')
        
        sns.heatmap(pivot_upload, annot=True, fmt='.0f', cmap='YlOrRd', ax=axes[1])
        axes[1].set_title('Upload Throughput Heatmap\n(Client Workers vs Server Workers)')
        
        plt.tight_layout()
        plt.savefig('performance_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def find_optimal_configurations(self):
        """Find optimal configurations"""
        print("=== OPTIMAL CONFIGURATIONS ===\n")
        
        # Best throughput overall
        best_throughput = self.df.loc[self.df['Throughput_Bytes_Per_Second'].idxmax()]
        print("Best Throughput Configuration:")
        print(f"Operation: {best_throughput['Operation']}")
        print(f"File Size: {best_throughput['Volume_MB']}MB")
        print(f"Client Workers: {best_throughput['Client_Workers']}")
        print(f"Server Workers: {best_throughput['Server_Workers']}")
        print(f"Throughput: {best_throughput['Throughput_Bytes_Per_Second']:.2f} B/s")
        print(f"Success Rate: {best_throughput['Client_Successful']}/{best_throughput['Client_Successful'] + best_throughput['Client_Failed']}")
        print()
        
        # Best configurations by operation
        for operation in ['download', 'upload']:
            op_data = self.df[self.df['Operation'] == operation]
            best_op = op_data.loc[op_data['Throughput_Bytes_Per_Second'].idxmax()]
            
            print(f"Best {operation.title()} Configuration:")
            print(f"File Size: {best_op['Volume_MB']}MB")
            print(f"Client Workers: {best_op['Client_Workers']}")  
            print(f"Server Workers: {best_op['Server_Workers']}")
            print(f"Throughput: {best_op['Throughput_Bytes_Per_Second']:.2f} B/s")
            print(f"Time: {best_op['Total_Time_Seconds']:.2f}s")
            print()
        
        # Most reliable configurations (100% success rate)
        reliable = self.df[self.df['Client_Failed'] == 0]
        if not reliable.empty:
            print("Most Reliable High-Performance Configurations:")
            top_reliable = reliable.nlargest(5, 'Throughput_Bytes_Per_Second')
            for i, (_, row) in enumerate(top_reliable.iterrows(), 1):
                print(f"{i}. {row['Operation']} {row['Volume_MB']}MB - "
                      f"C{row['Client_Workers']}/S{row['Server_Workers']} - "
                      f"{row['Throughput_Bytes_Per_Second']:.2f} B/s")
    
    def export_summary_table(self, filename: str = 'stress_test_summary_table.csv'):
        """Export formatted summary table"""
        summary = self.df.copy()
        
        # Add calculated columns
        summary['Success_Rate'] = summary['Client_Successful'] / \
                                 (summary['Client_Successful'] + summary['Client_Failed'])
        summary['Throughput_MB_per_sec'] = summary['Throughput_Bytes_Per_Second'] / (1024 * 1024)
        
        # Select and rename columns for final table
        final_columns = {
            'No': 'Test_ID',
            'Operation': 'Operation',
            'Volume_MB': 'File_Size_MB',
            'Client_Workers': 'Client_Workers',
            'Server_Workers': 'Server_Workers', 
            'Total_Time_Seconds': 'Avg_Time_Per_Client_Sec',
            'Throughput_MB_per_sec': 'Throughput_MB_per_sec',
            'Client_Successful': 'Successful_Clients',
            'Client_Failed': 'Failed_Clients',
            'Server_Successful': 'Successful_Server_Ops',
            'Server_Failed': 'Failed_Server_Ops',
            'Success_Rate': 'Success_Rate'
        }
        
        summary_table = summary[list(final_columns.keys())].rename(columns=final_columns)
        summary_table.to_csv(filename, index=False, float_format='%.2f')
        print(f"Summary table exported to {filename}")

def main():
    """Main function to run analysis"""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python server_manager.py analyze <results_file.csv>")
        print("  python server_manager.py start-server <type> <port> <workers>")
        return
    
    command = sys.argv[1]
    
    if command == "analyze":
        if len(sys.argv) < 3:
            print("Please specify results CSV file")
            return
            
        csv_file = sys.argv[2]
        if not os.path.exists(csv_file):
            print(f"File {csv_file} not found")
            return
        
        analyzer = ResultsAnalyzer(csv_file)
        
        print("Generating analysis...")
        analyzer.find_optimal_configurations()
        analyzer.export_summary_table()
        
        print("\nGenerating plots...")
        analyzer.analyze_throughput_by_operation()
        analyzer.analyze_success_rates()
        analyzer.analyze_response_times()
        analyzer.generate_performance_heatmap()
        
        print("Analysis complete! Check the generated PNG files and CSV summary.")
    
    elif command == "start-server":
        if len(sys.argv) < 5:
            print("Usage: python server_manager.py start-server <type> <port> <workers>")
            return
        
        server_type = sys.argv[2]
        port = int(sys.argv[3])
        workers = int(sys.argv[4])
        
        manager = ServerManager()
        manager.start_server("test", server_type, port, workers)
        
        print(f"Server started on port {port}. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            manager.stop_all_servers()
            print("Server stopped.")

if __name__ == "__main__":
    main()