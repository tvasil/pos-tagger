(function () {
  "use strict";

  const fixtures = window.POS_DEMO_FIXTURES;
  const tagDefinitions = window.POS_TAGS;

  const form = document.querySelector("#sentence-form");
  const input = document.querySelector("#sentence-input");
  const status = document.querySelector("#form-status");
  const tokenList = document.querySelector("#token-list");
  const tokenCount = document.querySelector("#token-count");
  const legend = document.querySelector("#tag-legend");
  const inspectorTitle = document.querySelector("#inspector-title");
  const inspectorTag = document.querySelector("#inspector-tag");
  const inspectorSummary = document.querySelector("#inspector-summary");
  const featureGroups = document.querySelector("#feature-groups");
  const comparisonCards = document.querySelector("#comparison-cards");
  const liveForm = document.querySelector("#live-form");
  const liveInput = document.querySelector("#live-input");
  const liveStatus = document.querySelector("#live-status");
  const liveResults = document.querySelector("#live-results");
  const liveTokenList = document.querySelector("#live-token-list");
  const liveTokenCount = document.querySelector("#live-token-count");
  const evidenceTag = document.querySelector("#evidence-tag");
  const evidenceSummary = document.querySelector("#evidence-summary");
  const transitionEvidence = document.querySelector("#transition-evidence");
  const contributionList = document.querySelector("#contribution-list");

  let currentFixture = fixtures[0];
  let selectedIndex = currentFixture.focusToken;
  let browserModelPromise;
  let livePrediction;
  let liveSelectedIndex = 0;

  function normalizeSentence(value) {
    return value.trim().replace(/\s+/g, " ").replace(/\s+([.!?,;:])/g, "$1").toLowerCase();
  }

  function findFixture(value) {
    const normalized = normalizeSentence(value);
    return fixtures.find((fixture) => normalizeSentence(fixture.sentence) === normalized);
  }

  function extractFeatures(tokens, index) {
    const words = tokens.map((token) => token.text);
    const raw = window.PosCrf.extractTokenFeatures(words, index);
    const features = {
      lexical: [
        ["Original", raw.word],
        ["Lowercase", raw["word.lower"]],
        ["Prefix · 1", raw["pre-1"]],
        ["Prefix · 2", raw["pre-2"]],
        ["Prefix · 3", raw["pre-3"]],
        ["Suffix · 1", raw["suf-1"]],
        ["Suffix · 2", raw["suf-2"]],
        ["Suffix · 3", raw["suf-3"]]
      ],
      shape: [
        ["Structure", raw.structure],
        ["Title case", raw.is_title],
        ["All caps", raw.is_all_caps],
        ["All lowercase", raw.is_all_lower],
        ["Alphabetic", raw.is_only_alpha],
        ["Alphanumeric", raw.is_alphanumeric],
        ["Numeric", raw.is_numeric],
        ["Has hyphen", raw.has_hyphen],
        ["Capitals inside", raw.capitals_inside]
      ],
      position: [
        ["First token", raw.is_first],
        ["Last token", raw.is_last]
      ],
      context: []
    };

    if (index > 0) {
      features.context.push(
        ["Previous word", raw["prev.word"]],
        ["Previous lowercase", raw["prev.word.lower"]],
        ["Previous is title case", raw["prev.word.is_title"]],
        ["Previous is first", raw["prev.word.is_first"]]
      );
    }

    if (index < tokens.length - 1) {
      features.context.push(
        ["Next word", raw["next.word"]],
        ["Next lowercase", raw["next.word.lower"]],
        ["Next is title case", raw["next.word.is_title"]],
        ["Next is last", raw["next.word.is_last"]]
      );
    }

    return features;
  }

  function createTokenButton(token, index) {
    const definition = tagDefinitions[token.tag];
    const button = document.createElement("button");
    button.type = "button";
    button.className = `token token--${definition.color}`;
    button.dataset.index = String(index);
    button.setAttribute("aria-pressed", String(index === selectedIndex));
    button.setAttribute("aria-label", `${token.text}, ${definition.name}. Inspect features.`);
    button.innerHTML = `<span class="token-word"></span><span class="token-tag"></span>`;
    button.querySelector(".token-word").textContent = token.text;
    button.querySelector(".token-tag").textContent = token.tag;
    button.addEventListener("click", () => selectToken(index));
    return button;
  }

  function renderTokens() {
    tokenList.replaceChildren();
    currentFixture.tokens.forEach((token, index) => tokenList.append(createTokenButton(token, index)));
    tokenCount.textContent = `${currentFixture.tokens.length} tokens`;
  }

  function renderLegend() {
    const usedTags = [...new Set(currentFixture.tokens.map((token) => token.tag))];
    legend.replaceChildren();

    usedTags.forEach((tag) => {
      const definition = tagDefinitions[tag];
      const item = document.createElement("li");
      item.innerHTML = `<span class="legend-dot legend-dot--${definition.color}"></span><strong></strong><span></span>`;
      item.querySelector("strong").textContent = tag;
      item.querySelector("span:last-child").textContent = definition.name;
      legend.append(item);
    });
  }

  function formatFeatureValue(value) {
    if (typeof value !== "boolean") return value;
    return value ? "Yes" : "No";
  }

  function renderFeatureGroup(title, rows) {
    if (!rows.length) return null;
    const section = document.createElement("section");
    section.className = "feature-group";
    const heading = document.createElement("h4");
    heading.textContent = title;
    const list = document.createElement("dl");

    rows.forEach(([label, value]) => {
      const row = document.createElement("div");
      const term = document.createElement("dt");
      const description = document.createElement("dd");
      term.textContent = label;
      description.textContent = formatFeatureValue(value);
      if (typeof value === "boolean") description.className = value ? "value-yes" : "value-no";
      row.append(term, description);
      list.append(row);
    });

    section.append(heading, list);
    return section;
  }

  function renderInspector() {
    const token = currentFixture.tokens[selectedIndex];
    const definition = tagDefinitions[token.tag];
    const features = extractFeatures(currentFixture.tokens, selectedIndex);

    inspectorTitle.textContent = `“${token.text}”`;
    inspectorTag.textContent = `${token.tag} · ${definition.name}`;
    inspectorTag.className = `inspector-tag inspector-tag--${definition.color}`;
    inspectorSummary.textContent = `These observations are available to the CRF when it scores the tag sequence. They are not independent rules.`;
    featureGroups.replaceChildren();

    [
      ["Word identity", features.lexical],
      ["Word structure", features.shape],
      ["Sentence position", features.position],
      ["Neighboring words", features.context]
    ].forEach(([title, rows]) => featureGroups.append(renderFeatureGroup(title, rows)));
  }

  function selectToken(index) {
    selectedIndex = index;
    tokenList.querySelectorAll(".token").forEach((button) => {
      button.setAttribute("aria-pressed", String(Number(button.dataset.index) === selectedIndex));
    });
    renderInspector();
  }

  function renderFixture(fixture) {
    currentFixture = fixture;
    selectedIndex = fixture.focusToken;
    input.value = fixture.sentence;
    status.textContent = "";
    renderTokens();
    renderLegend();
    renderInspector();
  }

  function renderComparison() {
    comparisonCards.replaceChildren();
    fixtures.forEach((fixture) => {
      const token = fixture.tokens[fixture.focusToken];
      const definition = tagDefinitions[token.tag];
      const features = extractFeatures(fixture.tokens, fixture.focusToken);
      const previous = features.context.find(([label]) => label === "Previous word")[1];
      const next = features.context.find(([label]) => label === "Next word")[1];
      const card = document.createElement("article");
      card.className = "comparison-card";
      card.innerHTML = `
        <p class="comparison-sentence"></p>
        <div class="comparison-result">
          <span class="comparison-word"></span>
          <span class="comparison-arrow" aria-hidden="true">→</span>
          <span class="comparison-tag"></span>
        </div>
        <dl>
          <div><dt>Previous word</dt><dd class="previous-word"></dd></div>
          <div><dt>Next word</dt><dd class="next-word"></dd></div>
        </dl>
      `;
      card.querySelector(".comparison-sentence").textContent = fixture.sentence;
      card.querySelector(".comparison-word").textContent = token.text;
      card.querySelector(".comparison-tag").textContent = `${token.tag} · ${definition.name}`;
      card.querySelector(".comparison-tag").classList.add(`comparison-tag--${definition.color}`);
      card.querySelector(".previous-word").textContent = previous;
      card.querySelector(".next-word").textContent = next;
      comparisonCards.append(card);
    });
  }

  function loadBrowserModel() {
    if (!browserModelPromise) {
      browserModelPromise = fetch("model/crf-model.json")
        .then((response) => {
          if (!response.ok) throw new Error(`Model request failed with ${response.status}.`);
          return response.json();
        })
        .then(window.PosCrf.validateModel)
        .catch((error) => {
          browserModelPromise = undefined;
          throw error;
        });
    }
    return browserModelPromise;
  }

  function renderEvidence(model) {
    const token = livePrediction.tokens[liveSelectedIndex];
    const tag = livePrediction.tags[liveSelectedIndex];
    const definition = tagDefinitions[tag];
    const contributions = window.PosCrf.featureContributions(
      livePrediction.features[liveSelectedIndex],
      tag,
      model
    );

    evidenceTag.textContent = `${tag} · ${definition.name}`;
    evidenceTag.className = `inspector-tag inspector-tag--${definition.color}`;
    evidenceSummary.textContent = `Strongest state-feature weights supporting or opposing “${token}” as ${definition.name.toLowerCase()}.`;
    contributionList.replaceChildren();

    contributions.slice(0, 9).forEach(({ attribute, contribution }) => {
      const item = document.createElement("li");
      const name = document.createElement("code");
      const score = document.createElement("span");
      name.textContent = attribute;
      score.textContent = `${contribution >= 0 ? "+" : ""}${contribution.toFixed(3)}`;
      score.className = contribution >= 0 ? "score-positive" : "score-negative";
      item.append(name, score);
      contributionList.append(item);
    });

    if (liveSelectedIndex === 0) {
      transitionEvidence.textContent = "Sequence start · no previous-tag transition";
    } else {
      const previousTag = livePrediction.tags[liveSelectedIndex - 1];
      const transition = window.PosCrf.transitionContribution(previousTag, tag, model);
      transitionEvidence.replaceChildren();
      const label = document.createElement("span");
      const score = document.createElement("strong");
      label.textContent = `${previousTag} → ${tag} transition`;
      score.textContent = `${transition >= 0 ? "+" : ""}${transition.toFixed(3)}`;
      score.className = transition >= 0 ? "score-positive" : "score-negative";
      transitionEvidence.append(label, score);
    }
  }

  function selectLiveToken(index, model) {
    liveSelectedIndex = index;
    liveTokenList.querySelectorAll(".token").forEach((button) => {
      button.setAttribute("aria-pressed", String(Number(button.dataset.index) === index));
    });
    renderEvidence(model);
  }

  function renderLivePrediction(model) {
    liveTokenList.replaceChildren();
    livePrediction.tokens.forEach((text, index) => {
      const tag = livePrediction.tags[index];
      const definition = tagDefinitions[tag];
      const button = document.createElement("button");
      button.type = "button";
      button.className = `token token--${definition.color}`;
      button.dataset.index = String(index);
      button.setAttribute("aria-pressed", String(index === liveSelectedIndex));
      button.setAttribute("aria-label", `${text}, ${definition.name}. Inspect learned evidence.`);
      button.innerHTML = `<span class="token-word"></span><span class="token-tag"></span>`;
      button.querySelector(".token-word").textContent = text;
      button.querySelector(".token-tag").textContent = tag;
      button.addEventListener("click", () => selectLiveToken(index, model));
      liveTokenList.append(button);
    });
    liveTokenCount.textContent = `${livePrediction.tokens.length} tokens`;
    liveResults.hidden = false;
    renderEvidence(model);
  }

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const fixture = findFixture(input.value);
    if (fixture) {
      renderFixture(fixture);
      return;
    }

    status.textContent = "This prototype only has predictions for the two examples. Choose one to continue.";
    status.focus();
  });

  document.querySelectorAll(".example-button").forEach((button) => {
    button.addEventListener("click", () => renderFixture(findFixture(button.dataset.example)));
  });

  liveForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const tokens = window.PosCrf.tokenize(liveInput.value);
    if (tokens.length === 0) {
      liveStatus.textContent = "Enter a sentence to tag.";
      return;
    }
    if (tokens.length > 80) {
      liveStatus.textContent = "Keep this demo to 80 tokens or fewer.";
      return;
    }

    liveStatus.textContent = "Loading the exported CRF and decoding locally…";
    const startedAt = performance.now();
    try {
      const model = await loadBrowserModel();
      livePrediction = window.PosCrf.tagTokens(tokens, model);
      liveSelectedIndex = 0;
      renderLivePrediction(model);
      const elapsed = Math.max(1, Math.round(performance.now() - startedAt));
      liveStatus.textContent = `Tagged ${tokens.length} tokens locally in ${elapsed} ms. Model weights are cached for the next sentence.`;
    } catch (error) {
      liveResults.hidden = true;
      liveStatus.textContent = `The browser model could not be loaded: ${error.message}`;
    }
  });

  renderFixture(currentFixture);
  renderComparison();
})();
