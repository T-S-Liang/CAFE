(() => {
  let DATA = null;
  let currentCase = null;
  let currentSide = 'pos';
  let playIdx = 0;
  let playing = false;
  let playTimer = null;
  let speed = 1;
  let baseUrl = '';
  const SPEEDS = [0.5, 1, 2, 4];

  async function init() {
    baseUrl = document.querySelector('script[src*="demo.js"]').src.replace(/demo\.js.*/, '');
    const resp = await fetch(baseUrl + 'demo_data.json');
    DATA = await resp.json();
    renderCaseGrid();
    selectCase(DATA.cases[0].tid);
  }

  function renderCaseGrid() {
    const grid = document.getElementById('case-grid');
    grid.innerHTML = '';
    DATA.cases.forEach(c => {
      const card = document.createElement('div');
      card.className = 'case-card';
      card.dataset.tid = c.tid;
      card.innerHTML = `
        <span class="type-badge type-${c.type}">${c.type}</span>
        <h4>${c.title}</h4>
        <div class="prompts-preview">
          <span class="pos">+ ${c.pos_concept}</span>
          <span class="neg">- ${c.neg_concept}</span>
        </div>`;
      card.onclick = () => selectCase(c.tid);
      grid.appendChild(card);
    });
  }

  function selectCase(tid) {
    currentCase = DATA.cases.find(c => c.tid === tid);
    if (!currentCase) return;
    document.querySelectorAll('.case-card').forEach(el => {
      el.classList.toggle('active', el.dataset.tid === tid);
    });
    document.getElementById('demo-viewer').classList.add('visible');
    document.getElementById('viewer-title').textContent = currentCase.title;
    document.getElementById('viewer-image').src = baseUrl + currentCase.image;

    const btnPos = document.getElementById('btn-pos');
    const btnNeg = document.getElementById('btn-neg');
    btnPos.textContent = `+ ${currentCase.pos_concept}`;
    btnNeg.textContent = `- ${currentCase.neg_concept}`;

    currentSide = 'pos';
    btnPos.className = 'active-pos';
    btnNeg.className = '';
    resetPlayback();
  }

  function switchSide(side) {
    currentSide = side;
    const btnPos = document.getElementById('btn-pos');
    const btnNeg = document.getElementById('btn-neg');
    btnPos.className = side === 'pos' ? 'active-pos' : '';
    btnNeg.className = side === 'neg' ? 'active-neg' : '';
    resetPlayback();
  }

  function getSteps() {
    if (!currentCase) return [];
    return currentSide === 'pos' ? currentCase.pos_steps : currentCase.neg_steps;
  }

  function resetPlayback(autoplay) {
    stopPlay();
    playIdx = 0;
    document.getElementById('trace-scroll').innerHTML = '';
    updateControls();
    stepForward();
    if (autoplay !== false) {
      setTimeout(startPlay, 600);
    }
  }

  function buildStepEls(step) {
    const els = [];

    if (step.type === 'query') {
      const el = document.createElement('div');
      el.className = 'step-block step-query';
      el.innerHTML = `Segment this concept: <span class="concept">"${esc(step.concept)}"</span>`;
      els.push(el);
    }
    else if (step.type === 'agent') {
      if (step.think) {
        const el = document.createElement('div');
        el.className = 'step-block step-think';
        el.textContent = step.think;
        els.push(el);
      }
      if (step.tool_name) {
        const el = document.createElement('div');
        el.className = 'step-block step-tool';
        let paramsStr = '';
        if (step.tool_name === 'segment_phrase') paramsStr = `"${esc(step.tool_params.text_prompt || '')}"`;
        else if (step.tool_name === 'examine_masks') paramsStr = `[${(step.tool_params.mask_indices || []).join(', ')}]`;
        else if (step.tool_name === 'select_masks_and_return') paramsStr = `[${(step.tool_params.final_answer_masks || []).join(', ')}]`;
        el.innerHTML = `<span class="tool-icon">${toolIcon(step.tool_name)}</span><span class="tool-name">${esc(step.tool_name)}</span><span class="tool-params">${paramsStr}</span>`;
        els.push(el);
      }
      if (step.tool_name === 'select_masks_and_return') {
        const el = document.createElement('div');
        const isPos = currentSide === 'pos';
        const correctAction = isPos;
        el.className = `step-block step-outcome ${correctAction ? 'outcome-correct' : 'outcome-wrong'}`;
        el.innerHTML = `<span class="outcome-icon">${correctAction ? '&#10004;' : '&#10008;'}</span> Concept accepted ${correctAction ? '(correct)' : '(incorrect)'}`;
        els.push(el);
      } else if (step.tool_name === 'report_no_mask') {
        const el = document.createElement('div');
        const isNeg = currentSide === 'neg';
        const correctAction = isNeg;
        el.className = `step-block step-outcome ${correctAction ? 'outcome-correct' : 'outcome-wrong'}`;
        el.innerHTML = `<span class="outcome-icon">${correctAction ? '&#10004;' : '&#10008;'}</span> Concept rejected ${correctAction ? '(correct)' : '(incorrect)'}`;
        els.push(el);
      }
    }
    else if (step.type === 'tool_result') {
      const el = document.createElement('div');
      el.className = 'step-block step-result';
      let html = esc(step.text);
      if (step.images && step.images.length > 0) {
        html += '<div class="result-images">';
        step.images.forEach(src => {
          html += `<img src="${baseUrl}${esc(src)}" class="result-img" loading="lazy" onclick="this.classList.toggle('expanded')">`;
        });
        html += '</div>';
      } else if (step.has_image) {
        const n = step.n_images || 1;
        for (let i = 0; i < n; i++) html += ` <span class="img-placeholder">image</span>`;
      }
      el.innerHTML = html;
      els.push(el);
    }

    return els;
  }

  function stepForward() {
    const steps = getSteps();
    if (playIdx >= steps.length) { stopPlay(); return; }

    const scroll = document.getElementById('trace-scroll');
    const els = buildStepEls(steps[playIdx]);

    els.forEach(el => scroll.appendChild(el));
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        els.forEach(el => el.classList.add('visible'));
        scroll.scrollTop = scroll.scrollHeight;
      });
    });

    playIdx++;
    updateControls();
  }

  function stepBack() {
    if (playIdx <= 0) return;
    playIdx--;
    const scroll = document.getElementById('trace-scroll');
    const n = buildStepEls(getSteps()[playIdx]).length;
    for (let i = 0; i < n; i++) {
      if (scroll.lastChild) scroll.removeChild(scroll.lastChild);
    }
    updateControls();
  }

  function togglePlay() {
    playing ? stopPlay() : startPlay();
  }

  function startPlay() {
    playing = true;
    updateControls();
    playTick();
  }

  function playTick() {
    if (!playing) return;
    if (playIdx >= getSteps().length) { stopPlay(); return; }
    stepForward();
    playTimer = setTimeout(playTick, Math.max(500, 2000 / speed));
  }

  function stopPlay() {
    playing = false;
    if (playTimer) { clearTimeout(playTimer); playTimer = null; }
    updateControls();
  }

  function cycleSpeed() {
    const idx = SPEEDS.indexOf(speed);
    speed = SPEEDS[(idx + 1) % SPEEDS.length];
    document.getElementById('speed-btn').textContent = speed + 'x';
  }

  function updateControls() {
    const total = getSteps().length;
    document.getElementById('btn-back').disabled = playIdx <= 0;
    document.getElementById('btn-forward').disabled = playIdx >= total;
    document.getElementById('btn-play').textContent = playing ? '⏸' : '▶';
    document.getElementById('step-counter').textContent = `${playIdx}/${total}`;
    document.getElementById('progress-fill').style.width = (total > 0 ? (playIdx / total) * 100 : 0) + '%';
  }

  function toolIcon(name) {
    return { segment_phrase: '🔍', examine_masks: '🔬', select_masks_and_return: '✅', report_no_mask: '❌' }[name] || '🔧';
  }

  function esc(s) {
    const d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
  }

  window.demoSwitchSide = switchSide;
  window.demoStepBack = stepBack;
  window.demoStepForward = stepForward;
  window.demoTogglePlay = togglePlay;
  window.demoCycleSpeed = cycleSpeed;

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
