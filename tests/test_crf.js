"use strict";

const assert = require("node:assert/strict");
const PosCrf = require("../docs/crf.js");

assert.deepEqual(PosCrf.tokenize("I can't fish."), ["I", "can't", "fish", "."]);

const features = PosCrf.extractTokenFeatures(["I", "can", "fish", "."], 1);
assert.equal(features["word.lower"], "can");
assert.equal(features["prev.word"], "I");
assert.equal(features["next.word"], "fish");
assert.ok(PosCrf.encodeFeatures(features).some(([name]) => name === "word:can"));
assert.ok(PosCrf.encodeFeatures(features).some(([name]) => name === "is_all_lower"));
assert.ok(!PosCrf.encodeFeatures(features).some(([name]) => name === "is_first"));

const model = {
  formatVersion: 1,
  modelType: "linear-chain-crf",
  labels: ["NOUN", "VERB"],
  stateFeatures: {
    "word:fish": [0, 2, 1, 1],
    "prev.word:can": [1, 3]
  },
  transitions: [0, 0, 0, 0]
};

assert.deepEqual(PosCrf.tagTokens(["can", "fish"], model).tags, ["NOUN", "VERB"]);
console.log("Browser CRF unit checks passed.");
