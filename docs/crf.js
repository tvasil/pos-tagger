(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  root.PosCrf = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  const TOKEN_PATTERN = /[\p{L}\p{N}_]+(?:['’][\p{L}\p{N}_]+)*|[^\s]/gu;

  function tokenize(text) {
    return String(text).match(TOKEN_PATTERN) || [];
  }

  function characters(value) {
    return Array.from(value);
  }

  function isUpper(value) {
    return /\p{L}/u.test(value) && value === value.toLocaleUpperCase();
  }

  function isLower(value) {
    return /\p{L}/u.test(value) && value === value.toLocaleLowerCase();
  }

  function isTitle(value) {
    let hasCasedCharacter = false;
    let previousWasCased = false;

    for (const character of characters(value)) {
      const isUpperCharacter = /[\p{Lu}\p{Lt}]/u.test(character);
      const isLowerCharacter = /\p{Ll}/u.test(character);
      if (!isUpperCharacter && !isLowerCharacter) {
        previousWasCased = false;
        continue;
      }

      hasCasedCharacter = true;
      if ((!previousWasCased && !isUpperCharacter) || (previousWasCased && !isLowerCharacter)) {
        return false;
      }
      previousWasCased = true;
    }

    return hasCasedCharacter;
  }

  function getStructure(value) {
    return characters(value).map((character) => {
      if (/\p{N}/u.test(character)) return "d";
      if (/[\p{Lu}\p{Lt}]/u.test(character)) return "X";
      if (/\p{Ll}/u.test(character)) return "x";
      return character;
    }).join("");
  }

  function prefix(value, length) {
    return characters(value).slice(0, length).join("");
  }

  function suffix(value, length) {
    return characters(value).slice(-length).join("");
  }

  function extractTokenFeatures(sentence, index) {
    const word = sentence[index];
    const restOfWord = characters(word).slice(1).join("");
    const features = {
      word,
      "word.lower": word.toLocaleLowerCase(),
      is_first: index === 0,
      is_last: index === sentence.length - 1,
      is_title: isTitle(word),
      is_all_caps: isUpper(word),
      is_all_lower: isLower(word),
      is_alphanumeric: /^[\p{L}\p{N}]+$/u.test(word),
      is_only_alpha: /^\p{L}+$/u.test(word),
      "pre-1": prefix(word, 1),
      "pre-2": prefix(word, 2),
      "pre-3": prefix(word, 3),
      "suf-1": suffix(word, 1),
      "suf-2": suffix(word, 2),
      "suf-3": suffix(word, 3),
      has_hyphen: word.includes("-"),
      is_numeric: /^\p{N}+$/u.test(word),
      capitals_inside: restOfWord.toLocaleLowerCase() !== restOfWord,
      structure: getStructure(word)
    };

    if (index > 0) {
      const previous = sentence[index - 1];
      features["prev.word"] = previous;
      features["prev.word.lower"] = previous.toLocaleLowerCase();
      features["prev.word.is_title"] = isTitle(previous);
      features["prev.word.is_first"] = index - 1 === 0;
    }

    if (index < sentence.length - 1) {
      const next = sentence[index + 1];
      features["next.word"] = next;
      features["next.word.lower"] = next.toLocaleLowerCase();
      features["next.word.is_title"] = isTitle(next);
      features["next.word.is_last"] = index + 1 === sentence.length - 1;
    }

    return features;
  }

  function sent2features(sentence) {
    return sentence.map((_, index) => extractTokenFeatures(sentence, index));
  }

  function encodeFeatures(features) {
    const encoded = [];
    Object.entries(features).forEach(([name, value]) => {
      if (typeof value === "string") encoded.push([`${name}:${value}`, 1]);
      else if (typeof value === "boolean" && value) encoded.push([name, 1]);
      else if (typeof value === "number" && value !== 0) encoded.push([name, value]);
    });
    return encoded;
  }

  function validateModel(model) {
    if (!model || model.formatVersion !== 1 || model.modelType !== "linear-chain-crf") {
      throw new Error("Unsupported browser CRF model format.");
    }
    const labelCount = model.labels.length;
    if (model.transitions.length !== labelCount * labelCount) {
      throw new Error("The CRF transition matrix is incomplete.");
    }
    return model;
  }

  function stateScores(featureSequence, model) {
    const labelCount = model.labels.length;
    return featureSequence.map((features) => {
      const scores = new Float64Array(labelCount);
      encodeFeatures(features).forEach(([attribute, value]) => {
        const weights = model.stateFeatures[attribute];
        if (!weights) return;
        for (let index = 0; index < weights.length; index += 2) {
          scores[weights[index]] += weights[index + 1] * value;
        }
      });
      return scores;
    });
  }

  function viterbi(scores, model) {
    const tokenCount = scores.length;
    const labelCount = model.labels.length;
    if (tokenCount === 0) return [];

    let previousScores = Float64Array.from(scores[0]);
    const backPointers = [];

    for (let tokenIndex = 1; tokenIndex < tokenCount; tokenIndex += 1) {
      const currentScores = new Float64Array(labelCount);
      const tokenPointers = new Int32Array(labelCount);

      for (let currentLabel = 0; currentLabel < labelCount; currentLabel += 1) {
        let bestPreviousLabel = 0;
        let bestScore = -Infinity;
        for (let previousLabel = 0; previousLabel < labelCount; previousLabel += 1) {
          const transitionIndex = previousLabel * labelCount + currentLabel;
          const candidate = previousScores[previousLabel] + model.transitions[transitionIndex];
          if (candidate > bestScore) {
            bestScore = candidate;
            bestPreviousLabel = previousLabel;
          }
        }
        currentScores[currentLabel] = bestScore + scores[tokenIndex][currentLabel];
        tokenPointers[currentLabel] = bestPreviousLabel;
      }

      previousScores = currentScores;
      backPointers.push(tokenPointers);
    }

    let bestFinalLabel = 0;
    for (let labelIndex = 1; labelIndex < labelCount; labelIndex += 1) {
      if (previousScores[labelIndex] > previousScores[bestFinalLabel]) {
        bestFinalLabel = labelIndex;
      }
    }

    const labelPath = new Array(tokenCount);
    labelPath[tokenCount - 1] = bestFinalLabel;
    for (let tokenIndex = tokenCount - 1; tokenIndex > 0; tokenIndex -= 1) {
      labelPath[tokenIndex - 1] = backPointers[tokenIndex - 1][labelPath[tokenIndex]];
    }
    return labelPath.map((labelIndex) => model.labels[labelIndex]);
  }

  function tagTokens(tokens, model) {
    validateModel(model);
    const featureSequence = sent2features(tokens);
    const scores = stateScores(featureSequence, model);
    return {
      tokens,
      tags: viterbi(scores, model),
      features: featureSequence
    };
  }

  function tagSentence(sentence, model) {
    return tagTokens(tokenize(sentence), model);
  }

  function featureContributions(features, label, model) {
    const labelIndex = model.labels.indexOf(label);
    if (labelIndex < 0) return [];
    const contributions = [];

    encodeFeatures(features).forEach(([attribute, value]) => {
      const weights = model.stateFeatures[attribute];
      if (!weights) return;
      for (let index = 0; index < weights.length; index += 2) {
        if (weights[index] !== labelIndex) continue;
        const weight = weights[index + 1];
        contributions.push({ attribute, value, weight, contribution: value * weight });
      }
    });

    return contributions.sort((left, right) =>
      Math.abs(right.contribution) - Math.abs(left.contribution)
    );
  }

  function transitionContribution(previousLabel, currentLabel, model) {
    const previousIndex = model.labels.indexOf(previousLabel);
    const currentIndex = model.labels.indexOf(currentLabel);
    if (previousIndex < 0 || currentIndex < 0) return 0;
    return model.transitions[previousIndex * model.labels.length + currentIndex];
  }

  return {
    encodeFeatures,
    extractTokenFeatures,
    featureContributions,
    getStructure,
    isLower,
    isTitle,
    isUpper,
    sent2features,
    tagSentence,
    tagTokens,
    tokenize,
    transitionContribution,
    validateModel,
    viterbi
  };
});
