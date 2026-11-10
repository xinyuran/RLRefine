import { validateKeywordPayload } from './contract.mjs';

const source = document.getElementById('source');
const payloadInput = document.getElementById('payload');
const exampleSelect = document.getElementById('example');
const status = document.getElementById('status');
const highlight = document.getElementById('highlight');
const chips = document.getElementById('keywords');
const errorsList = document.getElementById('errors');
const exampleSource = '包装很好，屏幕清晰。';
const examples = {
  valid: { source: exampleSource, payload: { keywords: [['原文评价包装', '包装', 0.96], ['原文描述屏幕', '屏幕', 0.94], ['原文描述清晰', '清晰', 0.9]] } },
  format: { source: exampleSource, payload: { keywords: [['原文评价包装', '包装', '0.96'], ['重复的对象词', '包装', 0.9], ['原文描述屏幕', '屏幕', 0.94]] } },
  source: { source: exampleSource, payload: { keywords: [['原文描述屏幕', '屏幕', 0.94], ['原文没有提到续航', '续航', 0.9]] } },
  historical: {
    source: '终于收到我需要的宝贝了，东西很好，价美物廉，谢谢掌柜的！说实在，这是我购物以来让我最满意的一次购物。无论是掌柜的态度还是对物品，我都非常满意。掌柜态度很专业热情，有问必答，回复也很快……收到的时候包装完整，宝贝比我想象中的还要好！下次需要的时候我还会再来，到时候麻烦掌柜给个优惠。',
    payload: { keywords: [['宝贝收到后包装完整', '宝贝', '0.95'], ['宝贝比我想象中的还要好', '宝贝', '0.95'], ['掌柜态度很专业热情', '态度', '0.90'], ['这是我购物以来让我最满意的一次购物', '购物', '0.75']] }
  }
};

function clearResults(message, state = 'pending') {
  status.textContent = message;
  status.dataset.state = state;
  for (const id of ['json-status', 'structure-status', 'source-status']) document.getElementById(id).textContent = '—';
  highlight.textContent = source.value;
  chips.replaceChildren();
  errorsList.replaceChildren();
}

function showMatches(payload) {
  const words = [...new Set((Array.isArray(payload?.keywords) ? payload.keywords : [])
    .filter(item => Array.isArray(item) && typeof item[1] === 'string' && item[1].length > 0)
    .map(item => item[1]))];
  const covered = new Uint8Array(source.value.length);
  for (const word of words) {
    const chip = document.createElement('span');
    const found = source.value.includes(word);
    chip.className = `keyword${found ? '' : ' missing'}`;
    chip.textContent = `${word}${found ? '' : ' · absent'}`;
    chips.append(chip);
    for (let index = source.value.indexOf(word); index !== -1; index = source.value.indexOf(word, index + 1)) {
      covered.fill(1, index, index + word.length);
    }
  }
  highlight.replaceChildren();
  for (let start = 0; start < source.value.length;) {
    let end = start + 1;
    while (end < source.value.length && covered[end] === covered[start]) end++;
    const segment = document.createElement(covered[start] ? 'mark' : 'span');
    segment.textContent = source.value.slice(start, end);
    highlight.append(segment);
    start = end;
  }
}

function validate() {
  clearResults('Checking…');
  if (!source.value.trim()) {
    clearResults('Enter a source review first.', 'invalid');
    return;
  }
  if (source.value.length > 10000 || payloadInput.value.length > 50000) {
    clearResults('Input is too long: use up to 10,000 source and 50,000 JSON characters.', 'invalid');
    return;
  }
  let payload;
  try {
    payload = JSON.parse(payloadInput.value);
  } catch {
    clearResults('JSON could not be parsed.', 'invalid');
    document.getElementById('json-status').textContent = 'Failed';
    const li = document.createElement('li');
    li.textContent = 'Use one JSON object with double-quoted strings and no Markdown fences.';
    errorsList.append(li);
    return;
  }
  const { valid, errors } = validateKeywordPayload(payload, source.value);
  status.textContent = valid ? 'Contract passed' : `Contract failed · ${errors.length} issue${errors.length === 1 ? '' : 's'}`;
  status.dataset.state = valid ? 'valid' : 'invalid';
  document.getElementById('json-status').textContent = 'Passed';
  const sourceErrors = errors.filter(error => /keyword_not_in_source|keyword_duplicate/.test(error));
  document.getElementById('structure-status').textContent = errors.length > sourceErrors.length ? 'Failed' : 'Passed';
  document.getElementById('source-status').textContent = sourceErrors.length ? 'Failed' : errors.length ? 'See details' : 'Passed';
  for (const error of errors.length ? errors : ['All explicit rules passed. Task quality still needs evaluation.']) {
    const li = document.createElement('li');
    li.textContent = error;
    errorsList.append(li);
  }
  showMatches(payload);
}

function loadExample() {
  const example = examples[exampleSelect.value];
  source.value = example.source;
  payloadInput.value = JSON.stringify(example.payload, null, 2);
  document.getElementById('provenance').textContent = exampleSelect.value === 'historical'
    ? 'Preserved excerpt from the previously published README (base model). Partial source/output; no inference is running and no score is calculated.'
    : 'Hand-written teaching example. Confidence values illustrate the format; they are not measured probabilities or live model results.';
  validate();
}

exampleSelect.addEventListener('change', loadExample);
document.getElementById('reset').addEventListener('click', loadExample);
document.getElementById('validate').addEventListener('click', validate);
for (const input of [source, payloadInput]) input.addEventListener('input', () => clearResults('Input changed · validate again.'));
loadExample();
