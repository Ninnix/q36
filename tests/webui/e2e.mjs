// Browser smoke test: node tests/webui/e2e.mjs  (needs playwright; set PLAYWRIGHT_PATH to its index.mjs)
// Starts tests/webui_mock_server.py, drives the UI, writes screenshots to $SHOTS (default /tmp/webui-shots).
import { spawn } from 'node:child_process';
import { mkdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const { chromium } = await import(process.env.PLAYWRIGHT_PATH || 'playwright');
const here = path.dirname(fileURLToPath(import.meta.url));
const port = 8099;
const base = `http://127.0.0.1:${port}`;
const shots = process.env.SHOTS || '/tmp/webui-shots';
mkdirSync(shots, { recursive: true });

const server = spawn('python3', [path.join(here, '..', 'webui_mock_server.py'), String(port)], { stdio: 'inherit' });
await new Promise((r) => setTimeout(r, 600));

let failures = 0;
const check = (name, ok, extra = '') => { console.log(`${ok ? 'ok  ' : 'FAIL'} ${name}${ok ? '' : ' ' + extra}`); if (!ok) failures++; };

const browser = await chromium.launch();
const ctx = await browser.newContext({ viewport: { width: 1280, height: 800 } });
const page = await ctx.newPage();
const errors = [];
page.on('pageerror', (e) => errors.push('pageerror: ' + e.message));
page.on('console', (m) => { if (m.type() === 'error') errors.push('console: ' + m.text()); });

async function send(text) {
  await page.fill('.composer-input', text);
  await page.keyboard.press('Enter');
}
const waitIdle = () => page.waitForSelector('.send-btn:not(.stop)', { timeout: 15000 });

try {
  await page.goto(base + '/', { waitUntil: 'networkidle' });
  await page.waitForSelector('.composer-input');
  check('model badge shows mock model', (await page.textContent('.model-name')).includes('mock-q36'));
  await page.screenshot({ path: shots + '/01-empty.png' });

  await send('hello world');
  await page.waitForSelector('.msg.assistant');
  await waitIdle();
  check('user bubble rendered', (await page.textContent('.msg.user .bubble')).includes('hello world'));
  check('assistant has bold', await page.locator('.msg.assistant .md.content strong').count() > 0);
  check('code block rendered', await page.locator('.msg.assistant .codeblock').count() >= 2);
  check('table rendered', await page.locator('.msg.assistant table').count() === 1);
  check('math rendered', await page.locator('.msg.assistant .math').count() >= 1);
  check('think block exists', await page.locator('.msg.assistant .think').count() === 1);
  check('stats shown', (await page.locator('.msg-stats').first().textContent()).length > 0);
  check('title derived', (await page.textContent('.conv-title')).includes('hello'));
  check('chat in sidebar', await page.locator('.chat-item').count() === 1);
  await page.screenshot({ path: shots + '/02-chat.png' });

  // second turn must resend reasoning_content (KV reuse) and history
  await send('second');
  await waitIdle();
  const reqs = await (await fetch(base + '/__requests')).json();
  const last = reqs[reqs.length - 1];
  check('history resent', last.messages.filter((m) => m.role === 'assistant').length === 1);
  check('reasoning_content resent', typeof last.messages.find((m) => m.role === 'assistant').reasoning_content === 'string');
  check('stream usage requested', last.stream_options && last.stream_options.include_usage === true);

  // HTML preview inside sandboxed iframe
  const prev = page.locator('.msg.assistant').first().locator('.code-btn[data-action=preview]').first();
  await prev.click();
  await page.waitForSelector('.preview-frame');
  const frame = page.frameLocator('.preview-frame');
  await frame.locator('#hi').waitFor({ timeout: 5000 });
  check('html preview renders', (await frame.locator('#hi').textContent()) === 'Preview OK');
  await page.screenshot({ path: shots + '/03-preview.png' });
  await page.keyboard.press('Escape');
  check('modal closed by Esc', await page.locator('.overlay').count() === 0);

  // regenerate creates sibling and branch nav
  const lastAsst = page.locator('.msg.assistant').last();
  await lastAsst.hover();
  await lastAsst.locator('.icon-btn[title="Regenerate"]').first().click();
  await waitIdle();
  check('branch nav after regenerate', await page.locator('.branch-nav').count() >= 1);

  // edit user message -> new branch
  const firstUser = page.locator('.msg.user').first();
  await firstUser.hover();
  await firstUser.locator('.icon-btn[title="Edit"], .icon-btn[title="Изменить"]').first().click();
  await page.waitForSelector('.edit-area');
  await page.fill('.edit-area', 'edited hello');
  await page.keyboard.press('Control+Enter');
  await page.waitForSelector('.msg.assistant');
  await waitIdle();
  check('edit replaced message', (await page.textContent('.msg.user .bubble')).includes('edited hello'));
  check('only one exchange visible after edit', await page.locator('.msg.user').count() === 1);

  // stop
  await send('SLOW please');
  await page.waitForSelector('.send-btn.stop');
  await page.waitForTimeout(400);
  await page.click('.send-btn.stop');
  await waitIdle();
  check('stopped message kept', (await page.locator('.msg.assistant').last().textContent()).includes('word'));

  // context exceeded error
  await send('TOOLONG');
  await waitIdle();
  check('context error shown', await page.locator('.msg-error').count() >= 1);

  // search
  await page.fill('.search-input', 'nomatch-xyz');
  await page.waitForTimeout(400);
  check('search filters list', await page.locator('.chat-item').count() === 0);
  await page.fill('.search-input', 'edited');
  await page.waitForTimeout(400);
  check('search matches', await page.locator('.chat-item').count() === 1);
  await page.fill('.search-input', '');

  // settings + theme + language
  await page.keyboard.press('Control+,');
  await page.waitForSelector('.settings-dialog');
  await page.screenshot({ path: shots + '/04-settings.png' });
  for (const tab of ['chat', 'sampling', 'data', 'about']) {
    await page.click(`.tab[data-tab=${tab}]`);
    check('tab ' + tab, await page.locator('.tab-panel').count() === 1);
  }
  await page.click('.tab[data-tab=general]');
  await page.locator('.settings-dialog select').nth(1).selectOption('dark');
  check('dark theme applied', (await page.getAttribute('html', 'data-theme')) === 'dark');
  await page.locator('.settings-dialog select').nth(0).selectOption('ru');
  await page.waitForTimeout(200);
  check('russian applied', (await page.textContent('.new-chat')).match(/[А-Яа-я]/) !== null);
  await page.keyboard.press('Escape');
  await page.screenshot({ path: shots + '/05-dark-ru.png' });

  // attachment via file chooser (text file)
  await page.locator('input[type=file]').first().setInputFiles({ name: 'note.txt', mimeType: 'text/plain', buffer: Buffer.from('file body here') });
  await page.waitForSelector('.att-card');
  await send('read the file');
  await waitIdle();
  const r2 = await (await fetch(base + '/__requests')).json();
  const lastReq = r2[r2.length - 1];
  const lastUser = lastReq.messages.filter((m) => m.role === 'user').pop();
  check('text attachment sent', JSON.stringify(lastUser).includes('file body here'));

  // persistence across reload
  await page.reload({ waitUntil: 'networkidle' });
  await page.waitForSelector('.chat-item');
  check('chat persisted', await page.locator('.chat-item').count() >= 1);
  await page.locator('.chat-item').first().click();
  await page.waitForSelector('.msg.user');
  check('messages reload', await page.locator('.msg').count() >= 2);

  // mobile
  await page.setViewportSize({ width: 390, height: 780 });
  await page.waitForTimeout(300);
  await page.screenshot({ path: shots + '/06-mobile.png' });
  await page.click('.sidebar-toggle');
  await page.waitForTimeout(300);
  await page.screenshot({ path: shots + '/07-mobile-drawer.png' });
} catch (e) {
  console.log('EXCEPTION', e.message);
  failures++;
  await page.screenshot({ path: shots + '/zz-failure.png' }).catch(() => {});
}

const real = errors.filter((e) => !/status of 400/.test(e));
check('no page errors', real.length === 0, '\n' + real.join('\n'));
await browser.close();
server.kill();
console.log(failures ? `${failures} failure(s)` : 'all passed');
process.exit(failures ? 1 : 0);
