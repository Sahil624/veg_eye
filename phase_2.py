from ultralytics.utils.benchmarks import benchmark
import itertools
import pandas as pd
import os
import argparse


def benchmark_permutations(model_path, data_path, formats=None, imgsz=640, device='cpu'):
    # Default formats
    if formats is None:
        formats = ['onnx', 'openvino', 'ncnn']
    
    half_options = [True, False]
    int8_options = [True, False]
    
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    output_path = f"{model_name}_benchmark_results.csv"
    
    # Create DataFrame with columns (returned by ultralytics.utils.benchmarks.benchmark)
    columns = ['Model', 'Format', 'Half Precision', 'INT8', 'Size (MB)', 
              'Status❔', 'Inference time (ms/im)', 'FPS']
    df = pd.DataFrame(columns=columns)
    
    # Save incrementally function
    def save_result(result):
        nonlocal df
        # Append to DataFrame
        df = pd.concat([df, pd.DataFrame([result])], ignore_index=True)
        # Save to CSV after each result
        df.to_csv(output_path, index=False)
    
    # Process benchmark results
    def process_result(benchmark_df, model, format_type, half, int8):
        if benchmark_df.empty:
            return None
            
        row = benchmark_df.iloc[0]
        result = {
            'Model': model,
            'Format': format_type,
            'Half Precision': half,
            'INT8': int8
        }
        
        # Copy all columns from benchmark result
        for col in benchmark_df.columns:
            if col != 'Format':
                result[col] = row[col]
        
        save_result(result)
        return result

    # GPU benchmark
    try:
        y = benchmark(
            model=model_path,
            data=data_path,
            imgsz=imgsz,
            device='0',
            verbose=True,
            format='-'
        )
        process_result(y, model_name, 'pytorch-gpu', False, False)
    except Exception as e:
        print(f"Error benchmarking on GPU: {e}")

    # CPU benchmarks for each permutation
    for half, int8 in itertools.product(half_options, int8_options):
        try:
            y = benchmark(
                model=model_path,
                data=data_path,
                imgsz=imgsz,
                device=device,
                half=half,
                int8=int8,
                verbose=True,
                format='-'
            )
            process_result(y, model_name, 'pytorch-cpu', half, int8)
        except Exception as e:
            print(f"Error benchmarking source model with half={half}, int8={int8}: {e}")
    
    # Format permutations
    for fmt, half, int8 in itertools.product(formats, half_options, int8_options):
        try:
            y = benchmark(
                model=model_path,
                data=data_path,
                imgsz=imgsz,
                device=device,
                format=fmt,
                half=half,
                int8=int8,
                verbose=True
            )
            process_result(y, model_name, fmt, half, int8)
        except Exception as e:
            print(f"Error benchmarking {fmt} with half={half}, int8={int8}: {e}")
    
    # Final calculations
    if not df.empty:
        if 'Size (MB)' in df.columns and 'Inference time (ms/im)' in df.columns:
            df['Size/Speed Ratio'] = df['Size (MB)'] / df['Inference time (ms/im)']
        
        # Sort and save final version
        if 'Inference time (ms/im)' in df.columns:
            df = df.sort_values('Inference time (ms/im)')
            df.to_csv(output_path, index=False)
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Benchmark YOLO models on CPU')
    parser.add_argument('--models', nargs='+', required=True, help='Paths to model files')
    parser.add_argument('--data', type=str, default=None, help='Path to validation data YAML file')
    
    args = parser.parse_args()

    benchmark_permutations(args.models[0], args.data)

if __name__ == "__main__":
    main()