// Interface strings.  English is the fallback; add a language by adding a
// dictionary here and an entry to LANGUAGES.

export const LANGUAGES = [
  { code: 'en', name: 'English' },
  { code: 'ru', name: 'Русский' },
];

const en = {
  close: 'Close', ok: 'OK', cancel: 'Cancel', save: 'Save', delete: 'Delete', edit: 'Edit', copy: 'Copy', copied: 'Copied',
  copyFailed: 'Copy failed', more: 'More', remove: 'Remove', auto: 'System', light: 'Light', dark: 'Dark', enabled: 'Enabled', disabled: 'Disabled',
  newChat: 'New chat', searchChats: 'Search chats', chats: 'Chats', noChats: 'No chats yet', noResults: 'Nothing found', searchResults: 'Results',
  untitled: 'New conversation', generating: 'Generating…', toggleSidebar: 'Toggle sidebar', toggleTheme: 'Toggle theme',
  group_pinned: 'Pinned', group_today: 'Today', group_yesterday: 'Yesterday', group_week: 'Previous 7 days', group_month: 'Previous 30 days', group_older: 'Older',
  rename: 'Rename', pin: 'Pin', unpin: 'Unpin', deleteChat: 'Delete chat', exportMarkdown: 'Export as Markdown', exportJson: 'Export as JSON',
  confirmDeleteChat: 'Delete “{title}”? This cannot be undone.', confirmDeleteAll: 'Delete all chats stored in this browser? This cannot be undone.',
  confirmDeleteTurn: 'Delete this message and everything after it?', confirmDeleteReply: 'Delete this reply?', allDeleted: 'All chats deleted',
  chatNotFound: 'Chat not found',

  welcome: 'How can I help you today?', welcomeModel: 'Chatting with {model}', welcomeSub: 'Ask anything, attach files, or paste an image.',
  sug1: 'Explain how a transformer language model generates text, step by step',
  sug2: 'Write a Python script that renames files by their modification date',
  sug3: 'Give me a plan for learning Rust in 30 days',
  sug4: 'Compare SQLite and PostgreSQL for a small web app',

  placeholder: 'Message…', send: 'Send', stop: 'Stop', attach: 'Attach files or images', hintEnter: 'Enter to send, Shift+Enter for a new line',
  hintCtrlEnter: 'Ctrl+Enter to send, Enter for a new line', tooManyFiles: 'Too many attachments',
  attachError_noVision: 'Images need the server to run with --vision ({name})', attachError_unsupported: 'Unsupported file type: {name}',
  attachError_tooLarge: '{name} is too large (limit 2 MB)', attachError_binary: '{name} looks like a binary file', attachError_decode: 'Could not read {name}',
  attachment: 'Attachment', viewAttachment: 'View attachment',
  scrollToBottom: 'Scroll to bottom', contextUsage: 'Context window {pct}% used',

  roleUser: 'You', roleAssistant: 'Assistant', thinking: 'Thinking', thought: 'Thought process', thoughtFor: 'Thought for {time}',
  prevBranch: 'Previous version', nextBranch: 'Next version', regenerate: 'Regenerate', saveSubmit: 'Save & send', showMore: 'Show more', showLess: 'Show less',
  stopped: 'Generation stopped.', hitLimit: 'The reply was cut off at the token limit.', busyHint: 'Wait for the reply to finish, or stop it first.',
  preview: 'Preview', wrap: 'Wrap', tokPerSec: 'tok/s', tokens: 'tokens', prompt: 'prompt', cached: 'cached', firstToken: 'first token',
  errContext: 'The conversation ({prompt} tokens) does not fit the context window ({ctx} tokens). Delete older messages or start a new chat.',
  errNetwork: 'Cannot reach the server. Check that q36-server is running.',

  serverOnline: 'Server online', serverOffline: 'Cannot reach the server', systemPrompt: 'System prompt',
  systemChatHint: 'Applies to this chat. Changing it makes the server re-read the whole conversation.', systemNewHint: 'Default system prompt for new chats.',
  systemPlaceholder: 'You are a helpful assistant…',
  storageError: 'Could not save to browser storage', storageMemoryWarn: 'Browser storage is unavailable: chats will be lost when you close this tab.',
  importFailed: 'Could not import {name}', imported: 'Imported {n} chat(s)',

  settings: 'Settings', shortcuts: 'Keyboard shortcuts',
  tab_general: 'General', tab_chat: 'Chat', tab_sampling: 'Sampling', tab_data: 'Data', tab_about: 'Server',
  language: 'Language', theme: 'Theme', fontSize: 'Font size', wideLayout: 'Wide layout', enterToSend: 'Enter sends the message', showStats: 'Show generation statistics',
  defaultSystem: 'Default system prompt', systemHint: 'Used when you start a new chat.',
  thinkingMode: 'Thinking', thinkingHint: 'How much the model reasons before answering. “Default” uses the server setting.',
  think_default: 'Default', think_off: 'Off', think_low: 'Low', think_medium: 'Medium', think_high: 'High', think_xhigh: 'Extra high', think_max: 'Max',
  thinkHint_default: 'server setting', thinkHint_off: 'answer directly', thinkHint_low: '', thinkHint_medium: '', thinkHint_high: '', thinkHint_xhigh: '', thinkHint_max: 'slowest',
  thinkingDisplay: 'Thinking block', thinkOpenStreaming: 'Open while thinking', thinkOpenAlways: 'Always open', thinkOpenNever: 'Always collapsed',
  sendReasoning: 'Send reasoning back to the server', sendReasoningHint: 'Keeps the server prompt cache valid between turns.',
  autoTitle: 'Generate titles with the model', autoTitleHint: 'Costs an extra request and resets the server prompt cache.',
  samplingHint: 'Leave a field empty to use the server default.', serverDefault: 'server default', resetDefaults: 'Reset to defaults',
  gen_temperature: 'Temperature', gen_top_p: 'Top P', gen_top_k: 'Top K', gen_min_p: 'Min P', gen_presence_penalty: 'Presence penalty',
  gen_frequency_penalty: 'Frequency penalty', gen_max_tokens: 'Max tokens', gen_seed: 'Seed', gen_stop: 'Stop sequences',
  stopPlaceholder: 'one per line', stopHint: 'Generation ends when one of these appears.',
  exportAll: 'Export all chats', importChats: 'Import chats', deleteAll: 'Delete all chats',
  dataHint: 'Chats are stored only in this browser (IndexedDB). Export them to back up or move to another browser.',
  model: 'Model', contextSize: 'Context size', vision: 'Image input', parallelSlots: 'Parallel slots', storage: 'Chat storage',
  storageLocal: 'This browser', storageMemory: 'Memory only (not saved)',
  scNewChat: 'New chat', scSidebar: 'Toggle sidebar', scFocusInput: 'Focus the message box', scSettings: 'Open settings', scSearch: 'Search chats',
  scSend: 'Send / new line', scSaveEdit: 'Save an edited message', scStop: 'Stop generating', scHelp: 'Show this help',
};

const ru = {
  close: 'Закрыть', ok: 'ОК', cancel: 'Отмена', save: 'Сохранить', delete: 'Удалить', edit: 'Изменить', copy: 'Копировать', copied: 'Скопировано',
  copyFailed: 'Не удалось скопировать', more: 'Ещё', remove: 'Убрать', auto: 'Как в системе', light: 'Светлая', dark: 'Тёмная', enabled: 'Включено', disabled: 'Выключено',
  newChat: 'Новый чат', searchChats: 'Поиск по чатам', chats: 'Чаты', noChats: 'Чатов пока нет', noResults: 'Ничего не найдено', searchResults: 'Результаты',
  untitled: 'Новый разговор', generating: 'Генерация…', toggleSidebar: 'Показать/скрыть панель', toggleTheme: 'Сменить тему',
  group_pinned: 'Закреплённые', group_today: 'Сегодня', group_yesterday: 'Вчера', group_week: 'Предыдущие 7 дней', group_month: 'Предыдущие 30 дней', group_older: 'Ранее',
  rename: 'Переименовать', pin: 'Закрепить', unpin: 'Открепить', deleteChat: 'Удалить чат', exportMarkdown: 'Экспорт в Markdown', exportJson: 'Экспорт в JSON',
  confirmDeleteChat: 'Удалить «{title}»? Это нельзя отменить.', confirmDeleteAll: 'Удалить все чаты, сохранённые в этом браузере? Это нельзя отменить.',
  confirmDeleteTurn: 'Удалить это сообщение и всё после него?', confirmDeleteReply: 'Удалить этот ответ?', allDeleted: 'Все чаты удалены',
  chatNotFound: 'Чат не найден',

  welcome: 'Чем могу помочь?', welcomeModel: 'Общение с {model}', welcomeSub: 'Задайте вопрос, прикрепите файлы или вставьте изображение.',
  sug1: 'Объясни по шагам, как трансформер генерирует текст',
  sug2: 'Напиши скрипт на Python, который переименовывает файлы по дате изменения',
  sug3: 'Составь план изучения Rust за 30 дней',
  sug4: 'Сравни SQLite и PostgreSQL для небольшого веб-приложения',

  placeholder: 'Сообщение…', send: 'Отправить', stop: 'Остановить', attach: 'Прикрепить файлы или изображения', hintEnter: 'Enter — отправить, Shift+Enter — новая строка',
  hintCtrlEnter: 'Ctrl+Enter — отправить, Enter — новая строка', tooManyFiles: 'Слишком много вложений',
  attachError_noVision: 'Для изображений запустите сервер с --vision ({name})', attachError_unsupported: 'Неподдерживаемый тип файла: {name}',
  attachError_tooLarge: '{name} слишком большой (лимит 2 МБ)', attachError_binary: '{name} похож на бинарный файл', attachError_decode: 'Не удалось прочитать {name}',
  attachment: 'Вложение', viewAttachment: 'Просмотр вложения',
  scrollToBottom: 'Вниз', contextUsage: 'Контекстное окно заполнено на {pct}%',

  roleUser: 'Вы', roleAssistant: 'Ассистент', thinking: 'Размышляет', thought: 'Ход рассуждений', thoughtFor: 'Размышлял {time}',
  prevBranch: 'Предыдущая версия', nextBranch: 'Следующая версия', regenerate: 'Сгенерировать заново', saveSubmit: 'Сохранить и отправить', showMore: 'Показать полностью', showLess: 'Свернуть',
  stopped: 'Генерация остановлена.', hitLimit: 'Ответ оборван на лимите токенов.', busyHint: 'Дождитесь окончания ответа или остановите его.',
  preview: 'Просмотр', wrap: 'Перенос', tokPerSec: 'ток/с', tokens: 'токенов', prompt: 'промпт', cached: 'из кэша', firstToken: 'первый токен',
  errContext: 'Разговор ({prompt} токенов) не помещается в контекстное окно ({ctx} токенов). Удалите старые сообщения или начните новый чат.',
  errNetwork: 'Нет связи с сервером. Проверьте, что q36-server запущен.',

  serverOnline: 'Сервер доступен', serverOffline: 'Нет связи с сервером', systemPrompt: 'Системный промпт',
  systemChatHint: 'Действует в этом чате. После изменения сервер заново прочитает весь разговор.', systemNewHint: 'Системный промпт по умолчанию для новых чатов.',
  systemPlaceholder: 'Ты — полезный ассистент…',
  storageError: 'Не удалось сохранить в хранилище браузера', storageMemoryWarn: 'Хранилище браузера недоступно: чаты пропадут после закрытия вкладки.',
  importFailed: 'Не удалось импортировать {name}', imported: 'Импортировано чатов: {n}',

  settings: 'Настройки', shortcuts: 'Горячие клавиши',
  tab_general: 'Общие', tab_chat: 'Чат', tab_sampling: 'Сэмплинг', tab_data: 'Данные', tab_about: 'Сервер',
  language: 'Язык', theme: 'Тема', fontSize: 'Размер шрифта', wideLayout: 'Широкая раскладка', enterToSend: 'Enter отправляет сообщение', showStats: 'Показывать статистику генерации',
  defaultSystem: 'Системный промпт по умолчанию', systemHint: 'Используется в новых чатах.',
  thinkingMode: 'Размышление', thinkingHint: 'Насколько долго модель рассуждает перед ответом. «По умолчанию» — настройка сервера.',
  think_default: 'По умолчанию', think_off: 'Выкл.', think_low: 'Низкое', think_medium: 'Среднее', think_high: 'Высокое', think_xhigh: 'Очень высокое', think_max: 'Максимум',
  thinkHint_default: 'как на сервере', thinkHint_off: 'сразу ответ', thinkHint_low: '', thinkHint_medium: '', thinkHint_high: '', thinkHint_xhigh: '', thinkHint_max: 'самое долгое',
  thinkingDisplay: 'Блок размышлений', thinkOpenStreaming: 'Открыт во время размышления', thinkOpenAlways: 'Всегда открыт', thinkOpenNever: 'Всегда свёрнут',
  sendReasoning: 'Отправлять рассуждения обратно на сервер', sendReasoningHint: 'Сохраняет кэш промпта на сервере между репликами.',
  autoTitle: 'Придумывать названия моделью', autoTitleHint: 'Лишний запрос; сбрасывает кэш промпта на сервере.',
  samplingHint: 'Пустое поле — значение по умолчанию на сервере.', serverDefault: 'по умолчанию', resetDefaults: 'Сбросить',
  gen_temperature: 'Температура', gen_top_p: 'Top P', gen_top_k: 'Top K', gen_min_p: 'Min P', gen_presence_penalty: 'Штраф присутствия',
  gen_frequency_penalty: 'Штраф частоты', gen_max_tokens: 'Макс. токенов', gen_seed: 'Seed', gen_stop: 'Стоп-последовательности',
  stopPlaceholder: 'по одной в строке', stopHint: 'Генерация прекращается, когда появляется одна из них.',
  exportAll: 'Экспорт всех чатов', importChats: 'Импорт чатов', deleteAll: 'Удалить все чаты',
  dataHint: 'Чаты хранятся только в этом браузере (IndexedDB). Экспортируйте их для резервной копии или переноса.',
  model: 'Модель', contextSize: 'Размер контекста', vision: 'Ввод изображений', parallelSlots: 'Параллельные слоты', storage: 'Хранение чатов',
  storageLocal: 'Этот браузер', storageMemory: 'Только в памяти (не сохраняется)',
  scNewChat: 'Новый чат', scSidebar: 'Показать/скрыть панель', scFocusInput: 'Фокус на поле ввода', scSettings: 'Открыть настройки', scSearch: 'Поиск по чатам',
  scSend: 'Отправить / новая строка', scSaveEdit: 'Сохранить изменённое сообщение', scStop: 'Остановить генерацию', scHelp: 'Показать эту справку',
};

export const DICTIONARIES = { en, ru };

let current = 'en';

export function resolveLang(pref) {
  if (pref && pref !== 'auto' && DICTIONARIES[pref]) return pref;
  const nav = (globalThis.navigator && navigator.language) || 'en';
  const short = nav.slice(0, 2).toLowerCase();
  return DICTIONARIES[short] ? short : 'en';
}

export function setLang(code) { current = DICTIONARIES[code] ? code : 'en'; }

export function t(key, vars) {
  let s = DICTIONARIES[current][key] ?? en[key] ?? key;
  if (vars) s = s.replace(/\{(\w+)\}/g, (m, k) => (k in vars ? vars[k] : m));
  return s;
}
