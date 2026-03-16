# -*- coding:utf-8 -*-
"""
Base Test Runner - Common functionality for CPU and NPU comparison tests
"""

import argparse
import os
import subprocess
import sys
import numpy as np

from common.utils import ensure_dir, compare_results


class BaseRunner:
    """Base class for CPU and NPU comparison runs"""
    
    def __init__(self, config):
        """Initialize runner
        
        Args:
            config: Configuration object with necessary parameters
        """
        self.config = config
        self.script_dir = os.path.dirname(os.path.abspath(sys.modules[self.__class__.__module__].__file__))
    
    def run_inference(self, run_type, num_runs=1, input_file=None, run_number=1):
        """Run inference on CPU or NPU for multiple runs in a single process.
        
        Args:
            run_type: 'CPU' or 'NPU'
            num_runs: Number of runs to execute
            input_file: Path to input data file (optional)
            run_number: Run number for output directory structure
        
        Returns:
            bool: True if inference succeeded
        """
        script_name = f"run_{run_type.lower()}.py"
        script_path = os.path.join(self.script_dir, script_name)

        # Build command with arguments
        cmd = [sys.executable, script_path, '--num-runs', str(num_runs), '--run-number', str(run_number)]
        if input_file:
            cmd.extend(['--input-file', input_file])

        print(f"\n{'=' * 60}")
        print(f"Running {run_type} inference ({num_runs} runs in single process)...")
        print(f"{'=' * 60}")

        result = subprocess.run(
            cmd,
            capture_output=False
        )

        return result.returncode == 0
    
    def compare_results(self, num_runs=1):
        """Compare all results from CPU and NPU runs.
        
        Args:
            num_runs: Number of runs to compare
        
        Returns:
            bool: True if all results match
        """
        # This method should be implemented by subclass
        raise NotImplementedError("Subclass must implement compare_results method")
    
    def generate_inputs(self, num_runs=1):
        """Generate input data for each run.
        
        Args:
            num_runs: Number of runs to generate inputs for
        
        Returns:
            tuple: (inputs, input_file)
        """
        # This method should be implemented by subclass
        raise NotImplementedError("Subclass must implement generate_inputs method")
    
    def main(self):
        """Run CPU and NPU runs and compare results."""
        parser = argparse.ArgumentParser(description=self.config.DESCRIPTION)
        parser.add_argument('--num-runs', type=int, default=self.config.DEFAULT_NUM_RUNS,
                            help='Number of runs in single process')
        parser.add_argument('--skip-cpu', action='store_true',
                            help='Skip CPU inference, only run NPU')
        parser.add_argument('--skip-npu', action='store_true',
                            help='Skip NPU inference, only run CPU')
        parser.add_argument('--cleanup', action='store_true', default=False,
                           help='Clean up temporary input/output files after test (default: False)')
        
        # Add subclass-specific arguments
        self.add_args(parser)
        
        args = parser.parse_args()
        
        self.print_config(args)
        
        # Subclass-specific initialization
        if not self.initialize(args):
            sys.exit(1)
        
        # Generate input data
        print("\n" + "=" * 60)
        print("Running CPU vs NPU comparison test")
        print("=" * 60)
        
        inputs, input_file = self.generate_inputs(args.num_runs)
        
        # Run CPU inference (multiple runs in single process)
        if not args.skip_cpu:
            if not self.run_inference("CPU", num_runs=args.num_runs, input_file=input_file, run_number=1):
                print(f"ERROR: CPU inference failed")
                sys.exit(1)

        # Run NPU inference (multiple runs in single process)
        if not args.skip_npu:
            if not self.run_inference("NPU", num_runs=args.num_runs, input_file=input_file, run_number=1):
                print(f"ERROR: NPU inference failed")
                sys.exit(1)
        
        # Print data file locations
        print("\n" + "=" * 60)
        print("Data files saved at:")
        print("=" * 60)
        print(f"  Input data: {self.config.INPUT_DATA_FILE}")
        print(f"  CPU outputs: {self.config.CPU_ALL_OUTPUTS_FILE}")
        if not args.skip_npu:
            print(f"  NPU outputs: {self.config.NPU_ALL_OUTPUTS_FILE}")
        
        # Comparison phase (only if both CPU and NPU ran)
        test_passed = None
        if not args.skip_cpu and not args.skip_npu:
            test_passed = self.compare_results(num_runs=args.num_runs)
        else:
            print(f"\n{'=' * 60}")
            print(f"⚠️  Skipped comparison (CPU or NPU was skipped)")
            print("=" * 60)
        
        # Final summary
        print("\n" + "=" * 60)
        print("FINAL SUMMARY")
        print("=" * 60)
        if test_passed is None:
            print(f"⚠️  Comparison was skipped")
        elif test_passed:
            print(f"✅ TEST PASSED")
        else:
            print(f"❌ TEST FAILED")
        print("=" * 60)
        
        # Cleanup if requested
        if args.cleanup:
            print("\nCleaning up temporary files...")
            files_to_remove = [
                self.config.INPUT_DATA_FILE,
            ]
            if not args.skip_cpu and os.path.exists(self.config.CPU_ALL_OUTPUTS_FILE):
                files_to_remove.append(self.config.CPU_ALL_OUTPUTS_FILE)
            if not args.skip_npu and os.path.exists(self.config.NPU_ALL_OUTPUTS_FILE):
                files_to_remove.append(self.config.NPU_ALL_OUTPUTS_FILE)
            
            for file_path in files_to_remove:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"  Removed: {file_path}")
    
    def add_args(self, parser):
        """Add subclass-specific arguments to parser.
        
        Args:
            parser: ArgumentParser object
        """
        # Override in subclass if needed
        pass
    
    def print_config(self, args):
        """Print configuration.
        
        Args:
            args: Parsed arguments
        """
        print(self.config.DESCRIPTION)
        print("=" * 50)
        print(f"Configuration:")
        print(f"  Number of runs: {args.num_runs}")
        print(f"  Skip CPU: {args.skip_cpu}")
        print(f"  Skip NPU: {args.skip_npu}")
    
    def initialize(self, args):
        """Subclass-specific initialization.
        
        Args:
            args: Parsed arguments
        
        Returns:
            bool: True if initialization succeeded
        """
        # Override in subclass if needed
        return True