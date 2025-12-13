# Troubleshooting

## Permission Error: "The requested file could not be read"

### Problem
You're getting an error like:
```
Error: The requested file could not be read, typically due to permission problems
```

### Cause
The `model.safetensors` file is open in your IDE or another program, causing Windows to lock the file.

### Solution

1. **Close the file in your IDE:**
   - Close the `model.safetensors` tab/window in your editor
   - Don't try to view/edit binary files in your IDE

2. **Check for other programs:**
   - Make sure no other programs are using the checkpoint directory
   - Close any file explorers that have the folder open

3. **Wait a moment:**
   - Sometimes Windows takes a moment to release file locks
   - Wait 5-10 seconds after closing the file

4. **Try again:**
   ```bash
   python convert_to_fp16.py --checkpoint-dir checkpoint_5000 --output-dir checkpoint_fp16
   ```

### Prevention
- Don't open `.safetensors` or `.bin` files in your IDE (they're binary files)
- Close file tabs before running scripts that need to read the files

