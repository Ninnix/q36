#!/usr/bin/env python3
"""Exercise session saves and exit prompts through a real Vulkan agent PTY."""
import argparse
import errno
import fcntl
import json
import os
from pathlib import Path
import pty
import re
import select
import signal
import struct
import subprocess
import termios
import time


ROOT = Path(__file__).resolve().parents[1]
ANSI = re.compile(rb'\x1b\[[0-?]*[ -/]*[@-~]|\x1b[78]')
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--binary', type=Path, default=ROOT / 'q36-agent')
parser.add_argument('--model', type=Path, default=ROOT / 'gguf/Qwen3.6-35B-A3B-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-v2-imatrix.gguf')
parser.add_argument('--vision', type=Path)
parser.add_argument('--web', action='store_true', help='Also test Google search and page visits; permits browser startup')
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=False)


def decode_trace(part, generated_only=False):
    output = bytearray()
    active = not generated_only
    for line in part.splitlines():
        if 'prefill sync done' in line:
            active = True
        elif 'generation finished' in line or 'tokens label=' in line:
            active = not generated_only
        match = re.search(r' token index=\d+ id=\d+ .* hex=([0-9a-f]*)$', line)
        if active and match:
            output.extend(bytes.fromhex(match[1]))
    return output.decode(errors='replace')


class Agent:
    def __init__(self, name):
        self.log = out / name
        self.log.mkdir()
        self.trace_path = self.log / 'trace'
        self.raw = bytearray()
        self.pending = b''
        self.capture = (self.log / 'terminal.ansi').open('wb')
        self.master, slave = pty.openpty()
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack('HHHH', 40, 120, 0, 0))
        cmd = [str(args.binary.resolve()), '-m', str(args.model.resolve()),
               '--vulkan', '--ssd-streaming', '--ctx', '8192', '--nothink',
               '--temp', '0', '--tokens', '2048', '--seed', '12345',
               '--chdir', str(out), '--trace', str(self.trace_path)]
        if args.vision:
            cmd += ['--vision', str(args.vision.resolve())]
        (self.log / 'command.json').write_text(json.dumps(cmd))
        self.proc = subprocess.Popen(cmd, cwd=ROOT, stdin=slave, stdout=slave,
                                     stderr=slave, start_new_session=True,
                                     env=dict(os.environ, TERM='xterm-256color',
                                              Q36_AGENT_CACHE_DIR=str(out / 'cache')))
        os.close(slave)

    def collect(self, seconds=0.2):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if not select.select([self.master], [], [], max(0, end-time.monotonic()))[0]:
                break
            try:
                data = os.read(self.master, 65536)
            except OSError as e:
                if e.errno == errno.EIO:
                    break
                raise
            if not data:
                break
            self.raw.extend(data)
            self.capture.write(data)
            self.capture.flush()
            self.pending += data
            while b'\x1b[6n' in self.pending:
                self.pending = self.pending.split(b'\x1b[6n', 1)[1]
                os.write(self.master, b'\x1b[1;1R')
            self.pending = self.pending[-8:]

    def plain(self):
        return ANSI.sub(b'', self.raw)

    def trace(self):
        return self.trace_path.read_text() if self.trace_path.exists() else ''

    def idle(self):
        states = re.findall(rb'ctx [^|\r\n]*\| ([A-Za-z]+)', self.plain())
        return bool(states) and states[-1] == b'idle'

    def wait(self, pred, label, timeout=240):
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            self.collect()
            if pred():
                print(label, round(time.monotonic()-start, 2), flush=True)
                return
            assert self.proc.poll() is None, f'agent exited {self.proc.returncode}: {label}'
        raise AssertionError('timeout: ' + label + '\n' + self.plain()[-2000:].decode(errors='replace'))

    def ready(self):
        self.wait(lambda: 'tokens label=initial_system_prompt' in self.trace() and self.idle(), 'ready')

    def send(self, text):
        os.write(self.master, b'\x1b[200~' + text.encode() + b'\x1b[201~\r')

    def turn(self, prompt, label):
        start = len(self.trace())
        output = len(self.raw)
        self.send(prompt)
        approved = False

        def done():
            nonlocal approved
            if args.web and not approved and b'visible browser. Allow?' in self.raw[output:]:
                os.write(self.master, b'y\r')
                approved = True
            return 'generation finished' in self.trace()[start:] and self.idle()

        self.wait(done, label)
        part = self.trace()[start:]
        assert 'Tool error:' not in decode_trace(part) and b'q36-agent: ' not in self.raw[output:], label
        return part

    def save(self):
        start = len(self.raw)
        self.send('/save')
        self.wait(lambda: b'saved session ' in self.raw[start:] or b'save failed:' in self.raw[start:], 'save')
        match = re.search(rb'saved session ([0-9a-f]{8})', self.raw[start:])
        assert match, self.plain()[-500:]
        return match[1].decode()

    def quiet_cpu(self, label):
        stat = Path(f'/proc/{self.proc.pid}/stat')
        if not stat.exists():
            return

        def ticks():
            fields = stat.read_text().split(') ', 1)[1].split()
            return int(fields[11]) + int(fields[12])

        before = ticks()
        start = time.monotonic()
        self.collect(3)
        cpu = 100 * (ticks()-before) / os.sysconf('SC_CLK_TCK') / (time.monotonic()-start)
        print(f'{label}: {cpu:.2f}% CPU', flush=True)
        assert cpu < 5, f'{label}: worker still active ({cpu:.2f}% CPU)'

    def exit_prompt(self):
        start = len(self.trace())
        self.send('Without tools, list the integers from 1 to 1000, one per line.')
        self.wait(lambda: 'prefill sync done' in self.trace()[start:] and not self.idle(), 'generation active')
        start = len(self.raw)
        os.write(self.master, b'\x04')
        self.wait(lambda: b'Save current session?' in self.raw[start:], 'EOF save prompt')
        self.quiet_cpu('at save prompt')
        return start

    def close(self):
        if self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait()
        self.collect()
        (self.log / 'terminal.txt').write_bytes(self.plain())
        self.capture.close()
        os.close(self.master)


agent = Agent('save')
try:
    agent.ready()
    agent.turn('Remember the code SESSION_ORCHID_42. Reply with exactly READY.', 'text chat')
    sha = agent.save()
    agent.quiet_cpu('after text save')
    if args.web:
        part = agent.turn('Use google_search to search for site:example.com Example Domain. '
                          'Then use visit_page to open https://example.com and report its heading. '
                          'Use both web tools, no shell commands.', 'web search and visit')
        text = decode_trace(part)
        assert '<function=google_search>' in text and '<function=visit_page>' in text
        assert re.search(r'\[Example Domain\]\(https://(?:example.com/|www.google.com/goto\?)', text)
        assert 'visit_page url=https://example.com' in text
        assert 'Example Domain' in decode_trace(part, True)
    start = agent.exit_prompt()
    os.write(agent.master, b'y\r')
    agent.wait(lambda: b'saved session ' in agent.raw[start:], 'save interrupted text chat')
    assert ('saved session ' + sha).encode() in agent.raw[start:]
    agent.wait(lambda: agent.proc.poll() is not None, 'clean exit', 30)
    assert agent.proc.returncode == 0
finally:
    agent.close()

agent = Agent('restore')
try:
    agent.ready()
    start = len(agent.raw)
    agent.send('/switch ' + sha)
    agent.wait(lambda: b'--- end history ---' in agent.raw[start:] and agent.idle(), 'restore session')
    part = agent.turn('What was the code I asked you to remember? Reply with the code only.', 'restored chat')
    assert 'SESSION_ORCHID_42' in decode_trace(part, True)
    assert agent.save() == sha
    if args.vision:
        part = agent.turn('Use view_image to read ' + str(ROOT / 'tests/vision-fixtures/text.png') +
                          '. Report the ticket code, seat, and gate. Do not use other tools.', 'image chat')
        assert '<function=view_image>' in decode_trace(part, True)
        assert all(word in decode_trace(part, True) for word in ('MINT', '731', '14A', 'C7'))
        start = agent.exit_prompt()
        os.write(agent.master, b'y\r')
        agent.wait(lambda: b'Continue anyway?' in agent.raw[start:], 'image save limitation')
        assert b'sessions containing images cannot be saved yet' in agent.raw[start:]
        assert b'model is busy' not in agent.raw[start:]
        agent.quiet_cpu('after image save failure')
        start = len(agent.raw)
        os.write(agent.master, b'n\r')
        agent.wait(lambda: b'q36-agent>' in agent.raw[start:] and agent.idle(), 'cancel exit')
        agent.turn('Without tools, reply with exactly CONTINUED.', 'continue image chat')
        start = len(agent.raw)
        agent.send('/new')
        agent.wait(lambda: b'Save current session?' in agent.raw[start:], 'new session prompt')
        os.write(agent.master, b'n\r')
        agent.wait(lambda: b'Agent, context' in agent.raw[start:] and agent.idle(), 'new session')
        agent.turn('Reply with exactly RESET_OK.', 'text chat after image reset')
        agent.save()
    agent.send('/quit')
    agent.wait(lambda: agent.proc.poll() is not None, 'quit saved session', 30)
    assert agent.proc.returncode == 0
finally:
    agent.close()
print('PASS: save, restore, interrupted exit, idle CPU' + (', image exit cancellation' if args.vision else '') +
      (', web search and visit' if args.web else ''), flush=True)
