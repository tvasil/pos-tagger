(function () {
  "use strict";

  const tagDefinitions = window.POS_TAGS;
  const form = document.querySelector("#tag-form");
  const input = document.querySelector("#sentence-input");
  const status = document.querySelector("#model-status");
  const results = document.querySelector("#results");
  const sequenceTrack = document.querySelector("#sequence-track");
  const legend = document.querySelector("#tag-legend");
  const stageTitle = document.querySelector("#stage-title");
  const stageDescription = document.querySelector("#stage-description");
  const stepButtons = [...document.querySelectorAll("#process-steps button")];
  const playButton = document.querySelector("#play-steps");
  const sequencePanel = document.querySelector(".sequence-panel");
  const inspectorTitle = document.querySelector("#inspector-title");
  const inspectorTag = document.querySelector("#inspector-tag");
  const previousToken = document.querySelector("#previous-token");
  const currentToken = document.querySelector("#current-token");
  const nextToken = document.querySelector("#next-token");
  const featureList = document.querySelector("#feature-list");
  const contributionList = document.querySelector("#contribution-list");
  const transitionEvidence = document.querySelector("#transition-evidence");
  const inspector = document.querySelector(".inspector");
  const dialog = document.querySelector("#crf-dialog");
  const coachmark = document.querySelector("#crf-coachmark");
  const openExplainer = document.querySelector("#open-explainer");

  const stageCopy = {
    1: ["1 · Tokenize", "Split the sentence into the units the model will label."],
    2: ["2 · Observe", "Read each token’s form, structure, position, and immediate neighbors."],
    3: ["3 · Score possible tags", "Score every label; the nodes show the chosen label’s state score."],
    4: ["4 · Connect the sequence", "Add learned transition weights between neighboring tag choices."],
    5: ["5 · Decode the best path", "Viterbi returns the highest-scoring complete tag sequence."]
  };

  let modelPromise;
  let model;
  let prediction;
  let selectedIndex = 0;
  let activeStep = 5;
  let animationTimer;

  function loadModel() {
    if (!modelPromise) {
      modelPromise = fetch("model/crf-model.json")
        .then((response) => {
          if (!response.ok) throw new Error(`Model request failed with ${response.status}.`);
          return response.json();
        })
        .then(window.PosCrf.validateModel)
        .catch((error) => {
          modelPromise = undefined;
          throw error;
        });
    }
    return modelPromise;
  }

  function setStatus(message, isError) {
    status.textContent = message;
    status.classList.toggle("is-error", Boolean(isError));
  }

  function signedScore(value) {
    return `${value >= 0 ? "+" : ""}${value.toFixed(3)}`;
  }

  function chosenStateScore(index) {
    const contributions = window.PosCrf.featureContributions(
      prediction.features[index],
      prediction.tags[index],
      model
    );
    return contributions.reduce((sum, item) => sum + item.contribution, 0);
  }

  function createTokenNode(word, tag, index) {
    const definition = tagDefinitions[tag];
    const button = document.createElement("button");
    const showTag = activeStep >= 3;
    const showTransition = activeStep >= 4 && index < prediction.tokens.length - 1;
    const isContextStep = activeStep === 2;

    button.type = "button";
    button.dataset.index = String(index);
    button.className = "token-node";
    if (activeStep >= 3) button.classList.add(`token--${definition.color}`);
    if (isContextStep && Math.abs(index - selectedIndex) === 1) button.classList.add("is-neighbor");
    if (isContextStep && Math.abs(index - selectedIndex) > 1) button.classList.add("is-muted");
    button.setAttribute("aria-pressed", String(index === selectedIndex));
    button.setAttribute("aria-label", `${word}, predicted ${definition.name}. Inspect model evidence.`);

    const wordLabel = document.createElement("span");
    wordLabel.className = "word";
    wordLabel.textContent = word;
    const tagLabel = document.createElement("span");
    tagLabel.className = "tag";
    tagLabel.textContent = showTag ? tag : "";
    button.append(wordLabel, tagLabel);

    if (activeStep === 2) {
      const dot = document.createElement("span");
      dot.className = "observation-dot";
      dot.setAttribute("aria-hidden", "true");
      button.append(dot);
    }

    const score = document.createElement("span");
    score.className = "node-score";
    score.textContent = activeStep === 3 ? `state ${signedScore(chosenStateScore(index))}` : "";
    button.append(score);

    if (showTransition) {
      const arrow = document.createElement("span");
      arrow.className = "transition-arrow";
      arrow.setAttribute("aria-hidden", "true");
      button.append(arrow);
    }

    button.addEventListener("click", () => selectToken(index));
    return button;
  }

  function renderSequence() {
    sequenceTrack.replaceChildren();
    prediction.tokens.forEach((word, index) => {
      sequenceTrack.append(createTokenNode(word, prediction.tags[index], index));
    });
  }

  function renderLegend() {
    legend.replaceChildren();
    [...new Set(prediction.tags)].forEach((tag) => {
      const definition = tagDefinitions[tag];
      const item = document.createElement("li");
      const dot = document.createElement("i");
      dot.className = `token--${definition.color}`;
      const abbreviation = document.createElement("strong");
      abbreviation.textContent = tag;
      const name = document.createElement("span");
      name.textContent = definition.name;
      item.append(dot, abbreviation, name);
      legend.append(item);
    });
  }

  function featureRows(features) {
    const rows = [
      ["Lowercase", features["word.lower"]],
      ["Structure", features.structure],
      ["Prefix", features["pre-3"]],
      ["Suffix", features["suf-3"]],
      ["Title case", features.is_title ? "yes" : "no"],
      ["All lowercase", features.is_all_lower ? "yes" : "no"]
    ];
    if (features["prev.word"] !== undefined) rows.push(["Previous word", features["prev.word"]]);
    if (features["next.word"] !== undefined) rows.push(["Next word", features["next.word"]]);
    return rows;
  }

  function readableAttribute(attribute) {
    const replacements = {
      "word.lower:": "lowercase = ",
      "prev.word.lower:": "previous lowercase = ",
      "next.word.lower:": "next lowercase = ",
      "prev.word:": "previous word = ",
      "next.word:": "next word = ",
      "structure:": "structure = ",
      "pre-1:": "prefix 1 = ",
      "pre-2:": "prefix 2 = ",
      "pre-3:": "prefix 3 = ",
      "suf-1:": "suffix 1 = ",
      "suf-2:": "suffix 2 = ",
      "suf-3:": "suffix 3 = ",
      "word:": "word = "
    };
    for (const [prefix, label] of Object.entries(replacements)) {
      if (attribute.startsWith(prefix)) return label + attribute.slice(prefix.length);
    }
    return attribute.replaceAll("_", " ");
  }

  function renderInspector() {
    const word = prediction.tokens[selectedIndex];
    const tag = prediction.tags[selectedIndex];
    const definition = tagDefinitions[tag];
    const features = prediction.features[selectedIndex];
    const contributions = window.PosCrf.featureContributions(features, tag, model).slice(0, 6);
    const maxMagnitude = Math.max(...contributions.map((item) => Math.abs(item.contribution)), 0.001);

    inspectorTitle.textContent = `“${word}”`;
    inspectorTag.textContent = `${tag} · ${definition.name}`;
    inspectorTag.className = `inspector-tag inspector-tag--${definition.color}`;
    previousToken.textContent = selectedIndex > 0 ? prediction.tokens[selectedIndex - 1] : "sentence start";
    currentToken.textContent = word;
    nextToken.textContent = selectedIndex < prediction.tokens.length - 1
      ? prediction.tokens[selectedIndex + 1]
      : "sentence end";

    featureList.replaceChildren();
    featureRows(features).forEach(([label, value]) => {
      const row = document.createElement("div");
      const term = document.createElement("dt");
      const description = document.createElement("dd");
      term.textContent = label;
      description.textContent = value;
      row.append(term, description);
      featureList.append(row);
    });

    contributionList.replaceChildren();
    contributions.forEach(({ attribute, contribution }) => {
      const item = document.createElement("li");
      const name = document.createElement("code");
      const track = document.createElement("span");
      const bar = document.createElement("i");
      const score = document.createElement("span");
      name.textContent = readableAttribute(attribute);
      name.title = attribute;
      track.className = "weight-track";
      bar.style.width = `${Math.max(5, Math.abs(contribution) / maxMagnitude * 100)}%`;
      if (contribution < 0) bar.className = "is-negative";
      track.append(bar);
      score.textContent = signedScore(contribution);
      score.className = contribution >= 0 ? "score-positive" : "score-negative";
      item.append(name, track, score);
      contributionList.append(item);
    });

    transitionEvidence.replaceChildren();
    if (selectedIndex === 0) {
      transitionEvidence.textContent = "Sequence start · no previous-tag transition";
    } else {
      const previousTag = prediction.tags[selectedIndex - 1];
      const transition = window.PosCrf.transitionContribution(previousTag, tag, model);
      const label = document.createElement("span");
      const score = document.createElement("strong");
      label.textContent = `${previousTag} → ${tag}`;
      score.textContent = signedScore(transition);
      score.className = transition >= 0 ? "score-positive" : "score-negative";
      transitionEvidence.append(label, score);
    }
  }

  function selectToken(index) {
    selectedIndex = index;
    renderSequence();
    renderInspector();
  }

  function setStep(step) {
    activeStep = step;
    inspector.dataset.step = String(step);
    sequencePanel.dataset.step = String(step);
    const [title, description] = stageCopy[step];
    stageTitle.textContent = title;
    stageDescription.textContent = description;
    stepButtons.forEach((button) => {
      const buttonStep = Number(button.dataset.step);
      if (buttonStep === step) button.setAttribute("aria-current", "step");
      else button.removeAttribute("aria-current");
      button.classList.toggle("is-complete", buttonStep < step);
    });
    renderSequence();
  }

  function stopAnimation() {
    if (animationTimer) window.clearTimeout(animationTimer);
    animationTimer = undefined;
    playButton.innerHTML = '<span aria-hidden="true">▶</span> Play steps';
  }

  function playSteps() {
    stopAnimation();
    let step = 1;
    playButton.textContent = "Playing…";
    const advance = () => {
      setStep(step);
      if (step < 5) {
        step += 1;
        animationTimer = window.setTimeout(advance, 850);
      } else {
        animationTimer = window.setTimeout(stopAnimation, 500);
      }
    };
    advance();
  }

  async function tagSentence(sentence, options = {}) {
    if (window.location.protocol === "file:") {
      results.hidden = true;
      setStatus("This demo cannot load its model from a file:// URL. Start the local server shown in the README, then open http://localhost:8000.", true);
      return;
    }

    const tokens = window.PosCrf.tokenize(sentence);
    if (!tokens.length) {
      setStatus("Enter a sentence to tag.", true);
      return;
    }
    if (tokens.length > 80) {
      setStatus("Keep this demo to 80 tokens or fewer.", true);
      return;
    }

    setStatus(model ? "Decoding locally…" : "Loading the exported model and decoding locally…");
    const startedAt = performance.now();
    try {
      model = model || await loadModel();
      prediction = window.PosCrf.tagTokens(tokens, model);
      const canIndex = prediction.tokens.findIndex((token) => token.toLowerCase() === "can");
      const shouldAnimate = options.animate && !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      selectedIndex = canIndex >= 0 ? canIndex : 0;
      results.hidden = false;
      renderLegend();
      renderInspector();
      setStep(shouldAnimate ? 1 : 5);
      const elapsed = Math.max(1, Math.round(performance.now() - startedAt));
      setStatus(`Tagged ${tokens.length} tokens locally in ${elapsed} ms.`);
      if (shouldAnimate) playSteps();
      document.querySelectorAll(".example-button").forEach((button) => {
        button.setAttribute("aria-pressed", String(button.dataset.example === sentence));
      });
    } catch (error) {
      results.hidden = true;
      setStatus(`The browser model could not be loaded: ${error.message}`, true);
    }
  }

  function openDialog(openMethodology) {
    dialog.showModal();
    if (openMethodology) {
      const details = dialog.querySelector(".methodology");
      details.open = true;
      details.scrollIntoView({ block: "nearest" });
    }
  }

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    stopAnimation();
    tagSentence(input.value);
  });

  document.querySelectorAll(".example-button").forEach((button) => {
    button.addEventListener("click", () => {
      input.value = button.dataset.example;
      stopAnimation();
      tagSentence(input.value, { animate: true });
    });
  });

  stepButtons.forEach((button) => {
    button.addEventListener("click", () => {
      stopAnimation();
      setStep(Number(button.dataset.step));
    });
  });

  playButton.addEventListener("click", playSteps);
  openExplainer.addEventListener("click", () => openDialog(false));
  document.querySelector("#coachmark-learn").addEventListener("click", () => openDialog(false));
  document.querySelector("#dismiss-coachmark").addEventListener("click", () => {
    coachmark.hidden = true;
    openExplainer.hidden = false;
    try { localStorage.setItem("crf-introduction-dismissed", "true"); } catch (error) { /* storage is optional */ }
  });
  dialog.addEventListener("click", (event) => {
    if (event.target === dialog) dialog.close();
  });

  try {
    coachmark.hidden = localStorage.getItem("crf-introduction-dismissed") === "true";
  } catch (error) {
    coachmark.hidden = false;
  }
  openExplainer.hidden = !coachmark.hidden;

  tagSentence(input.value);
})();
