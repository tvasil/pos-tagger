"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const PosCrf = require("../docs/crf.js");

const repositoryRoot = path.resolve(__dirname, "..");
const modelPath = process.argv[2] || path.join(repositoryRoot, "docs/model/crf-model.json");
const casesPath = process.argv[3] || path.join(repositoryRoot, "tests/browser_validation_cases.json");
const model = JSON.parse(fs.readFileSync(modelPath, "utf8"));
const cases = JSON.parse(fs.readFileSync(casesPath, "utf8"));

cases.forEach((validationCase) => {
  const result = PosCrf.tagSentence(validationCase.sentence, model);
  assert.deepEqual(result.tokens, validationCase.tokens, `${validationCase.sentence}: token mismatch`);
  assert.deepEqual(result.tags, validationCase.tags, `${validationCase.sentence}: tag mismatch`);
});

console.log(`Browser predictions match Python for ${cases.length} sentences.`);
