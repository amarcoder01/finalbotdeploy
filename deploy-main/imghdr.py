"""Mock imghdr module for Python 3.13 compatibility"""

# Simple mock implementation of imghdr for compatibility
# This provides the basic functionality needed by python-telegram-bot

def what(file, h=None):
    """Determine the type of image contained in a file or byte stream."""
    if hasattr(file, 'read'):
        # File-like object
        header = file.read(32)
        file.seek(0)  # Reset file position
    elif isinstance(file, (bytes, bytearray)):
        header = bytes(file[:32])
    else:
        # Assume it's a file path
        try:
            with open(file, 'rb') as f:
                header = f.read(32)
        except (OSError, IOError):
            return None
    
    # Basic image format detection
    if header.startswith(b'\xff\xd8\xff'):
        return 'jpeg'
    elif header.startswith(b'\x89PNG\r\n\x1a\n'):
        return 'png'
    elif header.startswith(b'GIF87a') or header.startswith(b'GIF89a'):
        return 'gif'
    elif header.startswith(b'RIFF') and b'WEBP' in header:
        return 'webp'
    elif header.startswith(b'BM'):
        return 'bmp'
    elif header.startswith(b'\x00\x00\x01\x00'):
        return 'ico'
    
    return None

# Additional functions that might be used
def test_jpeg(h, f):
    """JPEG test function"""
    return h.startswith(b'\xff\xd8\xff')

def test_png(h, f):
    """PNG test function"""
    return h.startswith(b'\x89PNG\r\n\x1a\n')

def test_gif(h, f):
    """GIF test function"""
    return h.startswith(b'GIF87a') or h.startswith(b'GIF89a')

def test_webp(h, f):
    """WebP test function"""
    return h.startswith(b'RIFF') and b'WEBP' in h

def test_bmp(h, f):
    """BMP test function"""
    return h.startswith(b'BM')

def test_ico(h, f):
    """ICO test function"""
    return h.startswith(b'\x00\x00\x01\x00')