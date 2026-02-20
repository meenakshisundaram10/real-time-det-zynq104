"""
Installation & Setup Helper Script
Validates environment and provides installation guidance
"""

import sys
import subprocess
from pathlib import Path

class InstallationHelper:
    """Helper for installation verification and setup"""
    
    @staticmethod
    def check_python_version():
        """Check if Python version is compatible"""
        version = sys.version_info
        print(f"[CHECK] Python Version: {version.major}.{version.minor}.{version.micro}")
        
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            print("[ERROR] Python 3.8+ required!")
            return False
        
        print("[✓] Python version OK")
        return True
    
    @staticmethod
    def check_package(package_name, import_name=None):
        """Check if package is installed"""
        import_name = import_name or package_name
        
        try:
            __import__(import_name)
            print(f"[✓] {package_name} installed")
            return True
        except ImportError:
            print(f"[✗] {package_name} NOT installed")
            return False
    
    @staticmethod
    def check_gpu():
        """Check if GPU is available"""
        try:
            import torch
            if torch.cuda.is_available():
                print(f"[✓] GPU Available: {torch.cuda.get_device_name(0)}")
                print(f"  CUDA Version: {torch.version.cuda}")
                return True
            else:
                print("[i] GPU Not Available (CPU mode)")
                return False
        except Exception as e:
            print(f"[✗] GPU Check Failed: {e}")
            return False
    
    @staticmethod
    def check_webcam():
        """Check if webcam is available"""
        try:
            import cv2
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                print(f"[✓] Webcam Available: {int(w)}x{int(h)}")
                cap.release()
                return True
            else:
                print("[✗] Webcam Not Available")
                return False
        except Exception as e:
            print(f"[✗] Webcam Check Failed: {e}")
            return False
    
    @staticmethod
    def check_disk_space():
        """Check available disk space"""
        try:
            import shutil
            disk_usage = shutil.disk_usage(".")
            free_gb = disk_usage.free / (1024**3)
            print(f"[✓] Free Disk Space: {free_gb:.2f} GB")
            
            if free_gb < 2:
                print("[⚠] Warning: Less than 2GB free (may need more for models)")
                return True
            return True
        except Exception as e:
            print(f"[✗] Disk Space Check Failed: {e}")
            return False
    
    @staticmethod
    def install_packages():
        """Install all required packages"""
        print("\n" + "="*60)
        print("Installing Required Packages...")
        print("="*60 + "\n")
        
        try:
            subprocess.run([
                sys.executable, '-m', 'pip', 'install',
                '-r', 'requirements.txt'
            ], check=True)
            print("\n[✓] All packages installed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"\n[✗] Installation failed: {e}")
            return False
    
    @staticmethod
    def run_diagnostics():
        """Run full system diagnostics"""
        print("\n" + "="*60)
        print("YOLOv8 PROJECT DIAGNOSTICS")
        print("="*60 + "\n")
        
        results = {
            'Python': InstallationHelper.check_python_version(),
            'OpenCV': InstallationHelper.check_package('opencv', 'cv2'),
            'PyTorch': InstallationHelper.check_package('torch'),
            'Ultralytics': InstallationHelper.check_package('ultralytics'),
            'GPU': InstallationHelper.check_gpu(),
            'Webcam': InstallationHelper.check_webcam(),
            'Disk Space': InstallationHelper.check_disk_space(),
        }
        
        print("\n" + "="*60)
        print("DIAGNOSTICS SUMMARY")
        print("="*60)
        
        passed = sum(1 for v in results.values() if v)
        total = len(results)
        
        print(f"\nPassed: {passed}/{total}")
        
        for item, result in results.items():
            status = "[✓]" if result else "[✗]"
            print(f"  {status} {item}")
        
        if passed == total:
            print("\n[✓] All checks passed! Ready to run detection.")
            return True
        else:
            print(f"\n[⚠] {total - passed} check(s) failed. See above for details.")
            return False


def main():
    """Main function"""
    
    print("\n")
    print("╔" + "═"*58 + "╗")
    print("║" + " "*58 + "║")
    print("║" + "YOLOv8 Real-time Object Detection Setup Helper".center(58) + "║")
    print("║" + " "*58 + "║")
    print("╚" + "═"*58 + "╝")
    print()
    
    # Create helper
    helper = InstallationHelper()
    
    # Check if requirements.txt exists
    if not Path('requirements.txt').exists():
        print("[✗] requirements.txt not found!")
        print("[i] Make sure you're in the yolo_detection directory")
        return
    
    print("Select an option:")
    print("1. Run Diagnostics (Check system)")
    print("2. Install Packages (From requirements.txt)")
    print("3. Full Setup (Diagnostics + Install)")
    print("4. Check GPU Only")
    print("5. Check Webcam Only")
    print("6. Exit")
    print()
    
    choice = input("Enter choice (1-6): ").strip()
    
    if choice == '1':
        helper.run_diagnostics()
    
    elif choice == '2':
        helper.install_packages()
        helper.run_diagnostics()
    
    elif choice == '3':
        helper.run_diagnostics()
        print()
        install = input("\nProceed with installation? (y/n): ").strip().lower()
        if install == 'y':
            helper.install_packages()
    
    elif choice == '4':
        print()
        helper.check_gpu()
    
    elif choice == '5':
        print()
        helper.check_webcam()
    
    elif choice == '6':
        print("Goodbye!")
    
    else:
        print("[✗] Invalid choice!")
    
    print()


if __name__ == "__main__":
    main()
