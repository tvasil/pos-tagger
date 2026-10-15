window.POS_DEMO_FIXTURES = [
  {
    sentence: "I can fish.",
    focusToken: 1,
    tokens: [
      { text: "I", tag: "PRON" },
      { text: "can", tag: "AUX" },
      { text: "fish", tag: "VERB" },
      { text: ".", tag: "PUNCT" }
    ]
  },
  {
    sentence: "A can of fish.",
    focusToken: 1,
    tokens: [
      { text: "A", tag: "DET" },
      { text: "can", tag: "NOUN" },
      { text: "of", tag: "ADP" },
      { text: "fish", tag: "NOUN" },
      { text: ".", tag: "PUNCT" }
    ]
  }
];

window.POS_TAGS = {
  ADP: { name: "Adposition", color: "blue" },
  AUX: { name: "Auxiliary", color: "violet" },
  DET: { name: "Determiner", color: "amber" },
  NOUN: { name: "Noun", color: "coral" },
  PRON: { name: "Pronoun", color: "cyan" },
  PUNCT: { name: "Punctuation", color: "slate" },
  VERB: { name: "Verb", color: "green" }
};
