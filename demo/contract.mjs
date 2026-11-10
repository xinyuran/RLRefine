// Browser equivalent of core.target_contract.validate_keyword_payload.
// Parity and shared limits are checked by tests/test_demo_contract.py.
export const MAX_KEYWORD_LENGTH = 4;
export const MAX_KEYWORDS = 15;
export const KEYWORD_TUPLE_SIZE = 3;

export function validateKeywordPayload(payload, source) {
  const errors = [];
  if (!payload || Array.isArray(payload) || typeof payload !== 'object' ||
      Object.keys(payload).length !== 1 || !Object.hasOwn(payload, 'keywords')) {
    return { valid: false, errors: ['payload_top_level_invalid'] };
  }
  const keywords = payload.keywords;
  if (!Array.isArray(keywords) || keywords.length < 1 || keywords.length > MAX_KEYWORDS) {
    return { valid: false, errors: ['keyword_count_invalid'] };
  }
  const seen = new Set();
  keywords.forEach((item, index) => {
    const prefix = `keywords[${index}]`;
    if (!Array.isArray(item) || item.length !== KEYWORD_TUPLE_SIZE) {
      errors.push(`${prefix}.shape_invalid`);
      return;
    }
    const [explanation, keyword, confidence] = item;
    // Match Python's str.isspace(), including its control-character whitespace.
    const whitespaceOnly = /^[\u0009-\u000d\u001c-\u0020\u0085\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]*$/u;
    if (typeof explanation !== 'string' || whitespaceOnly.test(explanation)) {
      errors.push(`${prefix}.explanation_invalid`);
    }
    if (typeof keyword !== 'string' || [...keyword].length < 1 || [...keyword].length > MAX_KEYWORD_LENGTH) {
      errors.push(`${prefix}.keyword_invalid`);
    } else {
      if (!source.includes(keyword)) errors.push(`${prefix}.keyword_not_in_source`);
      if (seen.has(keyword)) errors.push(`${prefix}.keyword_duplicate`);
      seen.add(keyword);
    }
    if (typeof confidence !== 'number' || !Number.isFinite(confidence) || confidence < 0 || confidence > 1) {
      errors.push(`${prefix}.confidence_invalid`);
    }
  });
  return { valid: errors.length === 0, errors };
}
