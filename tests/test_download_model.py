import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest


SOURCE = (Path(__file__).resolve().parents[1] / 'download_model.sh').read_text()
PAYLOAD = b'GGUF test weights\n'
FILES = dict(re.findall(r'^(\w+_FILE)="([^"]+)"$', SOURCE, re.M))
TARGETS = {
    'q2': ['Q2_FILE'],
    'q4': ['Q4_FILE'],
    '27b': ['DENSE_FILE'],
    '27b-q6': ['DENSE_Q6_FILE'],
    '27b-iq3-xxs': ['DENSE_SMALL_FILE'],
    'flash-next': ['FLASH_FILE', 'FLASH_NGRAM_FILE'],
    'qwen38-flash-next': ['FLASH_FILE', 'FLASH_NGRAM_FILE'],
    'qwen36-vision': ['VISION_FILE'],
    'qwen38-vision': ['DENSE_VISION_FILE'],
}


class DownloadModel(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='q36 download ')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.bin = self.root / 'bin'
        self.bin.mkdir()
        for name in ('mkdir', 'mv', 'ln', 'dirname', 'wc'):
            (self.bin / name).symlink_to(shutil.which(name))
        hash_tool = next((name for name in ('sha256sum', 'shasum') if shutil.which(name)), None)
        if hash_tool is None:
            self.skipTest('requires sha256sum or shasum')
        (self.bin / hash_tool).symlink_to(shutil.which(hash_tool))
        self.script = self.root / 'download_model.sh'
        # Exercise real hashing and file operations without multi-GB fixtures.
        source = re.sub(r'expected_bytes=\d+', f'expected_bytes={len(PAYLOAD)}', SOURCE)
        source = re.sub(r'expected_sha=[0-9a-f]{64}',
                        f'expected_sha={hashlib.sha256(PAYLOAD).hexdigest()}', source)
        self.script.write_text(source)
        curl = self.bin / 'curl'
        curl.write_text(f'#!{sys.executable}\n' + '''import os
from pathlib import Path
import sys
args = sys.argv[1:]
assert args[args.index('-C') + 1] == '-'
out = Path(args[args.index('-o') + 1])
with open(os.environ['CURL_LOG'], 'a') as log:
    log.write(args[-1] + '\\n')
payload = b'GGUF test weights\\n'
mode = os.environ.get('CURL_MODE')
if mode == 'interrupt':
    out.write_bytes(payload[:5])
    sys.exit(18)
if mode == 'short':
    payload = payload[:-1]
if mode == 'corrupt':
    payload = b'X' + payload[1:]
offset = out.stat().st_size if out.exists() else 0
with out.open('ab') as dst:
    dst.write(payload[offset:])
''')
        curl.chmod(0o755)
        self.out = self.root / 'weights'
        self.out.mkdir()
        self.log = self.root / 'curl.log'
        self.env = dict(os.environ, PATH=str(self.bin), HOME=str(self.root),
                        HF_TOKEN='', Q36_GGUF_DIR='weights', CURL_LOG=str(self.log))
        self.env.pop('CURL_MODE', None)
        self.link = self.root / 'q36moe.gguf'

    def run_model(self, target='q2', *options):
        return subprocess.run(['/bin/sh', str(self.script), target, *options],
                              env=self.env, capture_output=True, text=True)

    def assert_rejected(self, result, message):
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn(message, result.stderr)
        self.assertFalse(self.link.is_symlink())

    def test_all_targets_and_existing_files(self):
        for target, keys in TARGETS.items():
            with self.subTest(target=target):
                for path in self.out.iterdir():
                    path.unlink()
                self.link.unlink(missing_ok=True)
                result = self.run_model(target, '--token', 'test-token')
                self.assertEqual(result.returncode, 0, result.stderr)
                for key in keys:
                    self.assertEqual((self.out / FILES[key]).read_bytes(), PAYLOAD)
                if 'vision' in target:
                    self.assertFalse(self.link.is_symlink())
                    repo = ('unsloth/Qwen3.6-35B-A3B-GGUF' if target == 'qwen36-vision'
                            else 'unsloth/Qwen3.8-27B-GGUF')
                    self.assertEqual(self.log.read_text().splitlines()[-1],
                                     f'https://huggingface.co/{repo}/resolve/main/mmproj-F16.gguf')
                else:
                    self.assertEqual(self.link.resolve(), self.out / FILES[keys[0]])
                calls = self.log.read_text()
                result = self.run_model(target)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.count('Verifying SHA-256:'), len(keys))
                self.assertEqual(self.log.read_text(), calls)

    def test_bad_existing_files(self):
        out = self.out / FILES['Q2_FILE']
        for content, message in ((b'', 'Incorrect file size'),
                                 (PAYLOAD[:-1], 'Incorrect file size'),
                                 (b'X' + PAYLOAD[1:], 'Checksum mismatch')):
            with self.subTest(content=content):
                out.write_bytes(content)
                self.assert_rejected(self.run_model(), message)
                self.assertEqual(out.read_bytes(), content)
                self.assertFalse(self.log.exists())

    def test_bad_downloads_stay_partial(self):
        out = self.out / FILES['Q2_FILE']
        part = Path(str(out) + '.part')
        for mode, message in (('short', 'Incorrect file size'),
                              ('corrupt', 'Checksum mismatch')):
            with self.subTest(mode=mode):
                part.unlink(missing_ok=True)
                self.env['CURL_MODE'] = mode
                self.assert_rejected(self.run_model(), message)
                self.assertTrue(part.exists())
                self.assertFalse(out.exists())

    def test_interrupted_download_resumes(self):
        out = self.out / FILES['Q2_FILE']
        part = Path(str(out) + '.part')
        self.env['CURL_MODE'] = 'interrupt'
        self.assertEqual(self.run_model().returncode, 18)
        self.assertEqual(part.read_bytes(), PAYLOAD[:5])
        self.assertFalse(out.exists())
        self.assertFalse(self.link.is_symlink())
        del self.env['CURL_MODE']
        result = self.run_model()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(out.read_bytes(), PAYLOAD)
        self.assertFalse(part.exists())

    def test_corrupt_second_shard_preserves_model_link(self):
        first = self.out / FILES['FLASH_FILE']
        second = self.out / FILES['FLASH_NGRAM_FILE']
        first.write_bytes(PAYLOAD)
        second.write_bytes(b'X' + PAYLOAD[1:])
        self.link.symlink_to('previous.gguf')
        result = self.run_model('flash-next')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Checksum mismatch', result.stderr)
        self.assertEqual(os.readlink(self.link), 'previous.gguf')
        self.assertFalse(self.log.exists())

    def test_aria2_sidecar_blocks_download(self):
        (self.out / (FILES['Q2_FILE'] + '.aria2')).touch()
        self.assert_rejected(self.run_model(), 'incomplete aria2 download')
        self.assertFalse(self.log.exists())

    def test_shasum_fallback_and_missing_hash_tool(self):
        (self.bin / 'sha256sum').unlink(missing_ok=True)
        shasum = shutil.which('shasum')
        if shasum:
            if not (self.bin / 'shasum').exists():
                (self.bin / 'shasum').symlink_to(shasum)
            result = self.run_model()
            self.assertEqual(result.returncode, 0, result.stderr)
            (self.bin / 'shasum').unlink()
            self.link.unlink()
            self.log.unlink()
        self.assert_rejected(self.run_model(), 'requires sha256sum or shasum')
        self.assertFalse(self.log.exists())


if __name__ == '__main__':
    unittest.main()
