import subprocess
import sys

def main():
    try:
        # Run pytest through the current Python executable
        result = subprocess.run([sys.executable, '-m', 'pytest', 'tests/'], 
                              capture_output=True, 
                              text=True, 
                              encoding='utf-8')
        
        with open('pytest_log.txt', 'w', encoding='utf-8') as f:
            f.write("=== STDOUT ===\n")
            f.write(result.stdout)
            f.write("\n=== STDERR ===\n")
            f.write(result.stderr)
            
        print("Logged to pytest_log.txt")
        sys.exit(result.returncode)
    except Exception as e:
        with open('pytest_log.txt', 'w', encoding='utf-8') as f:
            f.write(f"Exception occurred:\n{e}")
        print("Logged exception to pytest_log.txt")
        sys.exit(1)

if __name__ == '__main__':
    main()
