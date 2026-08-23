// State Management
let currentMode = 'gini'; // 'gini' or 'entropy'

// DOM Elements
const toggleGini = document.getElementById('toggle-gini');
const toggleEntropy = document.getElementById('toggle-entropy');

const sliderParentA = document.getElementById('slider-parent-a');
const sliderParentB = document.getElementById('slider-parent-b');
const sliderSplitA = document.getElementById('slider-split-a');
const sliderSplitB = document.getElementById('slider-split-b');

const valParentA = document.getElementById('val-parent-a');
const valParentB = document.getElementById('val-parent-b');
const valSplitA = document.getElementById('val-split-a');
const valSplitB = document.getElementById('val-split-b');
const maxSplitA = document.getElementById('max-split-a');
const maxSplitB = document.getElementById('max-split-b');

// Node Counts and Metrics Display
const countParentA = document.getElementById('count-parent-a');
const countParentB = document.getElementById('count-parent-b');
const countLeftA = document.getElementById('count-left-a');
const countLeftB = document.getElementById('count-left-b');
const countRightA = document.getElementById('count-right-a');
const countRightB = document.getElementById('count-right-b');

const metricParent = document.getElementById('metric-parent');
const metricLeft = document.getElementById('metric-left');
const metricRight = document.getElementById('metric-right');
const weightedImpurityVal = document.getElementById('weighted-impurity-val');

const gainTypeLabel = document.getElementById('gain-type-label');
const gainMetricValue = document.getElementById('gain-metric-value');

// Visual Item Containers
const itemsParent = document.getElementById('items-parent');
const itemsLeft = document.getElementById('items-left');
const itemsRight = document.getElementById('items-right');

// Math Walkthrough Elements
const mathStepParentTitle = document.getElementById('math-step-parent-title');
const mathStepLeftTitle = document.getElementById('math-step-left-title');
const mathStepRightTitle = document.getElementById('math-step-right-title');
const mathStepGainTitle = document.getElementById('math-step-gain-title');

const formulaParent = document.getElementById('formula-parent');
const formulaLeft = document.getElementById('formula-left');
const formulaRight = document.getElementById('formula-right');
const formulaGain = document.getElementById('formula-gain');

const subParent = document.getElementById('sub-parent');
const subLeft = document.getElementById('sub-left');
const subRight = document.getElementById('sub-right');
const subWeighted = document.getElementById('sub-weighted');
const subGain = document.getElementById('sub-gain');

// SVG Connectors
const connectorsSvg = document.getElementById('connectors-svg');
const leftPath = document.getElementById('left-path');
const rightPath = document.getElementById('right-path');

// Math functions
function calculateGini(a, b) {
  const total = a + b;
  if (total === 0) return 0;
  const pA = a / total;
  const pB = b / total;
  return 1 - (pA * pA + pB * pB);
}

function calculateEntropy(a, b) {
  const total = a + b;
  if (total === 0) return 0;
  const pA = a / total;
  const pB = b / total;
  const termA = pA > 0 ? pA * Math.log2(pA) : 0;
  const termB = pB > 0 ? pB * Math.log2(pB) : 0;
  return -(termA + termB);
}

function log2Substitution(p) {
  if (p === 0) return "0";
  return `${p.toFixed(2)} * log2(${p.toFixed(2)})`;
}

// Render colored items in nodes
function renderItems(container, countA, countB) {
  container.innerHTML = '';
  // Append Class A (Good) dots
  for (let i = 0; i < countA; i++) {
    const dot = document.createElement('div');
    dot.className = 'dot class-a';
    dot.title = 'Class A (Good)';
    container.appendChild(dot);
  }
  // Append Class B (Bad) dots
  for (let i = 0; i < countB; i++) {
    const dot = document.createElement('div');
    dot.className = 'dot class-b';
    dot.title = 'Class B (Bad)';
    container.appendChild(dot);
  }
}

// Draw tree connection lines dynamically based on node coordinates
function updateTreeLines() {
  const parentNode = document.getElementById('node-parent');
  const leftNode = document.getElementById('node-left');
  const rightNode = document.getElementById('node-right');
  
  if (!parentNode || !leftNode || !rightNode) return;

  const svgRect = connectorsSvg.getBoundingClientRect();
  const parentRect = parentNode.getBoundingClientRect();
  const leftRect = leftNode.getBoundingClientRect();
  const rightRect = rightNode.getBoundingClientRect();

  // Find center positions relative to the SVG canvas
  const parentBottomX = (parentRect.left + parentRect.right) / 2 - svgRect.left;
  const parentBottomY = parentRect.bottom - svgRect.top;

  const leftTopX = (leftRect.left + leftRect.right) / 2 - svgRect.left;
  const leftTopY = leftRect.top - svgRect.top;

  const rightTopX = (rightRect.left + rightRect.right) / 2 - svgRect.left;
  const rightTopY = rightRect.top - svgRect.top;

  // Set control points for smooth bezier curves
  const controlY = parentBottomY + 20;

  leftPath.setAttribute('d', `M ${parentBottomX} ${parentBottomY} Q ${parentBottomX} ${controlY}, ${leftTopX} ${leftTopY}`);
  rightPath.setAttribute('d', `M ${parentBottomX} ${parentBottomY} Q ${parentBottomX} ${controlY}, ${rightTopX} ${rightTopY}`);
}

// Core calculation and UI update logic
function updateCalculator() {
  // Read inputs
  const pA = parseInt(sliderParentA.value);
  const pB = parseInt(sliderParentB.value);
  
  // Set limits on split sliders
  sliderSplitA.max = pA;
  sliderSplitB.max = pB;
  
  maxSplitA.textContent = pA;
  maxSplitB.textContent = pB;

  // Clamp split values if they exceed new parent maximums
  if (parseInt(sliderSplitA.value) > pA) {
    sliderSplitA.value = pA;
  }
  if (parseInt(sliderSplitB.value) > pB) {
    sliderSplitB.value = pB;
  }

  const sA = parseInt(sliderSplitA.value);
  const sB = parseInt(sliderSplitB.value);

  // Calculate Right Node counts
  const rA = pA - sA;
  const rB = pB - sB;

  // Update text values
  valParentA.textContent = pA;
  valParentB.textContent = pB;
  valSplitA.textContent = sA;
  valSplitB.textContent = sB;

  countParentA.textContent = pA;
  countParentB.textContent = pB;
  countLeftA.textContent = sA;
  countLeftB.textContent = sB;
  countRightA.textContent = rA;
  countRightB.textContent = rB;

  // Perform math calculations based on active metric mode
  let parentImp, leftImp, rightImp, weightedImp, gain;

  if (currentMode === 'gini') {
    parentImp = calculateGini(pA, pB);
    leftImp = calculateGini(sA, sB);
    rightImp = calculateGini(rA, rB);
    
    const totalParent = pA + pB;
    const totalLeft = sA + sB;
    const totalRight = rA + rB;
    
    weightedImp = totalParent > 0 ? (totalLeft / totalParent) * leftImp + (totalRight / totalParent) * rightImp : 0;
    gain = parentImp - weightedImp;

    // Update UI numbers
    metricParent.textContent = parentImp.toFixed(4);
    metricLeft.textContent = leftImp.toFixed(4);
    metricRight.textContent = rightImp.toFixed(4);
    weightedImpurityVal.textContent = weightedImp.toFixed(4);
    gainMetricValue.textContent = gain.toFixed(4);

    gainTypeLabel.textContent = "Gini Gain";

    // Update Math formulas representation
    mathStepParentTitle.textContent = "Parent Node Gini Impurity";
    mathStepLeftTitle.textContent = "Left Child Node Gini Impurity";
    mathStepRightTitle.textContent = "Right Child Node Gini Impurity";
    mathStepGainTitle.textContent = "Gini Gain (Impurity Reduction)";

    formulaParent.textContent = "Gini = 1 - (p_A^2 + p_B^2)";
    formulaLeft.textContent = "Gini = 1 - (p_A^2 + p_B^2)";
    formulaRight.textContent = "Gini = 1 - (p_A^2 + p_B^2)";

    // Update substitutions
    const totalP = pA + pB;
    if (totalP === 0) {
      subParent.textContent = "1 - (0^2 + 0^2) = 0.0000";
    } else {
      subParent.textContent = `1 - ((${pA}/${totalP})^2 + (${pB}/${totalP})^2) = 1 - (${(pA/totalP).toFixed(2)}^2 + ${(pB/totalP).toFixed(2)}^2) = ${parentImp.toFixed(4)}`;
    }

    const totalL = sA + sB;
    if (totalL === 0) {
      subLeft.textContent = "1 - (0^2 + 0^2) = 0.0000";
    } else {
      subLeft.textContent = `1 - ((${sA}/${totalL})^2 + (${sB}/${totalL})^2) = 1 - (${(sA/totalL).toFixed(2)}^2 + ${(sB/totalL).toFixed(2)}^2) = ${leftImp.toFixed(4)}`;
    }

    const totalR = rA + rB;
    if (totalR === 0) {
      subRight.textContent = "1 - (0^2 + 0^2) = 0.0000";
    } else {
      subRight.textContent = `1 - ((${rA}/${totalR})^2 + (${rB}/${totalR})^2) = 1 - (${(rA/totalR).toFixed(2)}^2 + ${(rB/totalR).toFixed(2)}^2) = ${rightImp.toFixed(4)}`;
    }

    subWeighted.textContent = totalP > 0 ? 
      `(${totalL}/${totalP}) * ${leftImp.toFixed(4)} + (${totalR}/${totalP}) * ${rightImp.toFixed(4)} = ${weightedImp.toFixed(4)}` : 
      "0.0000";

    subGain.textContent = `${parentImp.toFixed(4)} - ${weightedImp.toFixed(4)} = ${gain.toFixed(4)}`;

  } else {
    // Entropy mode
    parentImp = calculateEntropy(pA, pB);
    leftImp = calculateEntropy(sA, sB);
    rightImp = calculateEntropy(rA, rB);
    
    const totalParent = pA + pB;
    const totalLeft = sA + sB;
    const totalRight = rA + rB;
    
    weightedImp = totalParent > 0 ? (totalLeft / totalParent) * leftImp + (totalRight / totalParent) * rightImp : 0;
    gain = parentImp - weightedImp;

    // Update UI numbers
    metricParent.textContent = parentImp.toFixed(4);
    metricLeft.textContent = leftImp.toFixed(4);
    metricRight.textContent = rightImp.toFixed(4);
    weightedImpurityVal.textContent = weightedImp.toFixed(4);
    gainMetricValue.textContent = gain.toFixed(4);

    gainTypeLabel.textContent = "Information Gain";

    // Update Math formulas representation
    mathStepParentTitle.textContent = "Parent Node Entropy";
    mathStepLeftTitle.textContent = "Left Child Node Entropy";
    mathStepRightTitle.textContent = "Right Child Node Entropy";
    mathStepGainTitle.textContent = "Information Gain (Entropy Reduction)";

    formulaParent.textContent = "Entropy = - [p_A*log2(p_A) + p_B*log2(p_B)]";
    formulaLeft.textContent = "Entropy = - [p_A*log2(p_A) + p_B*log2(p_B)]";
    formulaRight.textContent = "Entropy = - [p_A*log2(p_A) + p_B*log2(p_B)]";

    const totalP = pA + pB;
    if (totalP === 0) {
      subParent.textContent = "0.0000";
    } else {
      const pA_val = pA / totalP;
      const pB_val = pB / totalP;
      subParent.textContent = `- [${log2Substitution(pA_val)} + ${log2Substitution(pB_val)}] = ${parentImp.toFixed(4)}`;
    }

    const totalL = sA + sB;
    if (totalL === 0) {
      subLeft.textContent = "0.0000";
    } else {
      const pA_val = sA / totalL;
      const pB_val = sB / totalL;
      subLeft.textContent = `- [${log2Substitution(pA_val)} + ${log2Substitution(pB_val)}] = ${leftImp.toFixed(4)}`;
    }

    const totalR = rA + rB;
    if (totalR === 0) {
      subRight.textContent = "0.0000";
    } else {
      const pA_val = rA / totalR;
      const pB_val = rB / totalR;
      subRight.textContent = `- [${log2Substitution(pA_val)} + ${log2Substitution(pB_val)}] = ${rightImp.toFixed(4)}`;
    }

    subWeighted.textContent = totalP > 0 ? 
      `(${totalL}/${totalP}) * ${leftImp.toFixed(4)} + (${totalR}/${totalP}) * ${rightImp.toFixed(4)} = ${weightedImp.toFixed(4)}` : 
      "0.0000";

    subGain.textContent = `${parentImp.toFixed(4)} - ${weightedImp.toFixed(4)} = ${gain.toFixed(4)}`;
  }

  // Render node dots
  renderItems(itemsParent, pA, pB);
  renderItems(itemsLeft, sA, sB);
  renderItems(itemsRight, rA, rB);

  // Redraw SVG connections
  setTimeout(updateTreeLines, 50);
}

// Event Listeners for inputs
sliderParentA.addEventListener('input', updateCalculator);
sliderParentB.addEventListener('input', updateCalculator);
sliderSplitA.addEventListener('input', updateCalculator);
sliderSplitB.addEventListener('input', updateCalculator);

// Event Listeners for Toggle Buttons
toggleGini.addEventListener('click', () => {
  if (currentMode !== 'gini') {
    currentMode = 'gini';
    toggleGini.classList.add('active');
    toggleGini.setAttribute('aria-selected', 'true');
    toggleEntropy.classList.remove('active');
    toggleEntropy.setAttribute('aria-selected', 'false');
    updateCalculator();
  }
});

toggleEntropy.addEventListener('click', () => {
  if (currentMode !== 'entropy') {
    currentMode = 'entropy';
    toggleEntropy.classList.add('active');
    toggleEntropy.setAttribute('aria-selected', 'true');
    toggleGini.classList.remove('active');
    toggleGini.setAttribute('aria-selected', 'false');
    updateCalculator();
  }
});

// Resize listener to keep tree connections correctly positioned
window.addEventListener('resize', updateTreeLines);

// Initial call
updateCalculator();
// Delayed secondary redraw to ensure browser layout has finished rendering cards
setTimeout(updateTreeLines, 200);
