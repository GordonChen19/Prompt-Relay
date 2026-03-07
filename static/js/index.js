window.HELP_IMPROVE_VIDEOJS = false;


$(document).ready(function() {
    // Check for click events on the navbar burger icon

    var options = {
			slidesToScroll: 1,
			slidesToShow: 1,
			loop: true,
			infinite: true,
			autoplay: true,
			autoplaySpeed: 5000,
    }

		// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);
	
    bulmaSlider.attach();

    const defaultModelOptions = [
      { key: 'wan2-2', label: 'Wan2.2' },
      { key: 'kling-2-6', label: 'Kling 2.6' },
      { key: 'sora2', label: 'Sora2' },
    ];

    const normalizeModelKey = (value) => {
      return String(value || '')
        .trim()
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, '-')
        .replace(/^-+|-+$/g, '');
    };

    const parseCardModelOptions = (card) => {
      const raw = card.dataset.modelConfig;
      if (!raw) {
        return defaultModelOptions;
      }

      try {
        const parsed = JSON.parse(raw);
        if (!Array.isArray(parsed)) {
          return defaultModelOptions;
        }

        const seen = new Set();
        const options = parsed
          .filter((item) => item && typeof item === 'object')
          .map((item) => {
            const key = normalizeModelKey(item.key || item.label);
            const label = String(item.label || item.key || '').trim();
            return { key, label };
          })
          .filter((item) => item.key && item.label)
          .filter((item) => {
            if (seen.has(item.key)) {
              return false;
            }
            seen.add(item.key);
            return true;
          });

        return options.length > 0 ? options : defaultModelOptions;
      } catch (error) {
        console.warn('Invalid model configuration on card:', error);
        return defaultModelOptions;
      }
    };

    const getCardModelSource = (card, modelKey) => {
      return card.getAttribute(`data-model-${modelKey}`);
    };

    const preloadedSources = new Set();

    const preloadMediaSource = (src) => {
      const normalizedSrc = String(src || '').trim();
      if (!normalizedSrc || preloadedSources.has(normalizedSrc)) {
        return;
      }
      preloadedSources.add(normalizedSrc);

      if (/\.(mp4|webm|ogg)(\?.*)?$/i.test(normalizedSrc)) {
        const preloadVideo = document.createElement('video');
        preloadVideo.preload = 'auto';
        preloadVideo.muted = true;
        preloadVideo.src = normalizedSrc;
        preloadVideo.load();
        return;
      }

      const preloadImage = new Image();
      preloadImage.decoding = 'async';
      preloadImage.src = normalizedSrc;
    };

    const warmSourcesWhenIdle = (sources) => {
      const run = () => {
        sources.forEach((source) => {
          preloadMediaSource(source);
        });
      };

      if ('requestIdleCallback' in window) {
        window.requestIdleCallback(run, { timeout: 1500 });
      } else {
        setTimeout(run, 150);
      }
    };

    document.querySelectorAll('.video-card').forEach((card) => {
      const mediaClip = card.querySelector('.media-clip');
      if (!mediaClip) {
        return;
      }

      const defaultSrc = mediaClip.getAttribute('src');
      if (!defaultSrc) {
        return;
      }

      const cardModelOptions = parseCardModelOptions(card);
      const modelSources = {};
      cardModelOptions.forEach((model) => {
        const modelSrc = getCardModelSource(card, model.key);
        modelSources[model.key] = modelSrc && modelSrc.trim() ? modelSrc.trim() : defaultSrc;
      });
      const uniqueModelSources = Array.from(
        new Set(
          Object.values(modelSources)
            .map((value) => String(value || '').trim())
            .filter(Boolean)
        )
      );
      warmSourcesWhenIdle(uniqueModelSources);

      const mediaWrap = card.querySelector('.media-wrap');
      if (!mediaWrap) {
        return;
      }

      const existingSwitch = card.querySelector('.model-switch');
      if (existingSwitch) {
        existingSwitch.remove();
      }

      const modelSwitch = document.createElement('div');
      modelSwitch.className = 'model-switch';
      modelSwitch.setAttribute('role', 'group');
      modelSwitch.setAttribute('aria-label', 'Select model output');

      cardModelOptions.forEach((model) => {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'model-button';
        button.dataset.model = model.key;
        button.textContent = model.label;
        modelSwitch.appendChild(button);
      });

      mediaWrap.insertAdjacentElement('afterend', modelSwitch);

      const modelButtons = Array.from(modelSwitch.querySelectorAll('.model-button'));
      const baseAlt = mediaClip.getAttribute('alt') || 'Video';

      const setActiveModel = (modelKey) => {
        const activeButton = modelButtons.find((button) => button.dataset.model === modelKey) || modelButtons[0];
        if (!activeButton) {
          return;
        }

        const nextSrc = modelSources[activeButton.dataset.model] || defaultSrc;
        preloadMediaSource(nextSrc);
        if (mediaClip.getAttribute('src') !== nextSrc) {
          mediaClip.setAttribute('src', nextSrc);
        }
        if (mediaClip.tagName.toLowerCase() === 'video') {
          mediaClip.load();
          const playPromise = mediaClip.play();
          if (playPromise && typeof playPromise.catch === 'function') {
            playPromise.catch(() => {});
          }
        } else {
          mediaClip.setAttribute('alt', `${baseAlt} (${activeButton.textContent.trim()})`);
        }

        modelButtons.forEach((button) => {
          button.classList.toggle('is-active', button === activeButton);
        });
      };

      modelButtons.forEach((button) => {
        button.addEventListener('click', () => {
          setActiveModel(button.dataset.model);
        });
        const warmButtonModel = () => {
          const nextSrc = modelSources[button.dataset.model] || defaultSrc;
          preloadMediaSource(nextSrc);
        };
        button.addEventListener('mouseenter', warmButtonModel);
        button.addEventListener('focus', warmButtonModel);
      });

      const preferredModel = normalizeModelKey(card.dataset.defaultModel || '');
      const fallbackModel = cardModelOptions[0] ? cardModelOptions[0].key : '';
      setActiveModel(preferredModel || fallbackModel);
    });

    const costVisualizer = document.getElementById('cost-visualizer');
    if (costVisualizer) {
      const segmentLengthInput = document.getElementById('cost-segment-length');
      const segmentLengthValue = document.getElementById('cost-segment-length-value');
      const midpointInput = document.getElementById('cost-midpoint');
      const queryInput = document.getElementById('cost-query-frame');
      const windowDerivedValue = document.getElementById('cost-window-derived');
      const windowReadout = document.getElementById('cost-window-readout');
      const sigmaDerivedValue = document.getElementById('cost-sigma-derived');
      const sigmaReadout = document.getElementById('cost-sigma-readout');
      const segmentInterval = document.getElementById('cost-segment-interval');
      const midpointValue = document.getElementById('cost-midpoint-value');
      const queryValue = document.getElementById('cost-query-frame-value');
      const zeroInterval = document.getElementById('cost-zero-interval');
      const selectedCost = document.getElementById('cost-selected-cost');
      const selectedAttention = document.getElementById('cost-selected-attention');
      const gridLayer = document.getElementById('cost-grid-lines');
      const segmentZone = document.getElementById('cost-segment-zone');
      const noPenaltyZone = document.getElementById('cost-no-penalty-zone');
      const attentionPath = document.getElementById('cost-attention-path');
      const queryLine = document.getElementById('cost-query-line');
      const queryPoint = document.getElementById('cost-query-point');
      const queryLabel = document.getElementById('cost-query-label');

      const required = [
        segmentLengthInput,
        segmentLengthValue,
        midpointInput,
        queryInput,
        windowDerivedValue,
        windowReadout,
        sigmaDerivedValue,
        sigmaReadout,
        segmentInterval,
        midpointValue,
        queryValue,
        zeroInterval,
        selectedCost,
        selectedAttention,
        gridLayer,
        segmentZone,
        noPenaltyZone,
        attentionPath,
        queryLine,
        queryPoint,
        queryLabel,
      ];

      if (required.every(Boolean)) {
        const epsilon = 1e-3;
        const frameMin = 0;
        const frameMax = 100;
        const chartWidth = 760;
        const chartHeight = 300;
        const padding = {
          top: 16,
          right: 18,
          bottom: 44,
          left: 58,
        };
        const plotWidth = chartWidth - padding.left - padding.right;
        const plotHeight = chartHeight - padding.top - padding.bottom;

        const xToPx = (frame) => {
          const domain = Math.max(frameMax - frameMin, 1);
          return padding.left + ((frame - frameMin) / domain) * plotWidth;
        };

        const yToPx = (attention) => {
          return padding.top + (1 - attention) * plotHeight;
        };

        const drawGrid = () => {
          const xTicks = [0, 20, 40, 60, 80, 100];
          const yTicks = [1.0, 0.75, 0.5, 0.25, 0.0];
          const parts = [];

          yTicks.forEach((tick) => {
            const y = yToPx(tick);
            parts.push(`<line class="cost-grid-line" x1="${padding.left}" y1="${y}" x2="${padding.left + plotWidth}" y2="${y}"></line>`);
            parts.push(`<text class="cost-grid-label" x="${padding.left - 9}" y="${y + 4}" text-anchor="end">${Math.round(tick * 100)}%</text>`);
          });

          xTicks.forEach((tick) => {
            const x = xToPx(tick);
            parts.push(`<line class="cost-grid-line" x1="${x}" y1="${padding.top}" x2="${x}" y2="${padding.top + plotHeight}"></line>`);
            parts.push(`<text class="cost-grid-label" x="${x}" y="${padding.top + plotHeight + 16}" text-anchor="middle">${tick}</text>`);
          });

          parts.push(`<line class="cost-axis-line" x1="${padding.left}" y1="${padding.top}" x2="${padding.left}" y2="${padding.top + plotHeight}"></line>`);
          parts.push(`<line class="cost-axis-line" x1="${padding.left}" y1="${padding.top + plotHeight}" x2="${padding.left + plotWidth}" y2="${padding.top + plotHeight}"></line>`);

          gridLayer.innerHTML = parts.join('');
        };

        const computePoint = (frame, midpoint, windowSize, sigma) => {
          const distance = Math.abs(frame - midpoint);
          const relu = Math.max(distance - windowSize, 0);
          const cost = (relu * relu) / (2 * sigma * sigma);
          return {
            cost: cost,
            attention: Math.exp(-cost),
          };
        };

        const formatFrame = (value) => {
          if (Number.isInteger(value)) {
            return String(value);
          }
          return value.toFixed(1);
        };

        const clamp = (value, min, max) => {
          return Math.min(max, Math.max(min, value));
        };

        const formatMetric = (value, digits) => {
          const fixed = value.toFixed(digits);
          return fixed.replace(/\.?0+$/, '');
        };

        const computeSigma = (segmentLength) => {
          return segmentLength / (2 * Math.log(1 / epsilon));
        };

        const computeWindow = (segmentLength) => {
          return segmentLength / 3;
        };

        const updateBounds = (segmentLength) => {
          const halfLength = segmentLength / 2;
          const midpointMin = frameMin + halfLength;
          const midpointMax = frameMax - halfLength;

          midpointInput.min = String(midpointMin);
          midpointInput.max = String(midpointMax);

          midpointInput.value = String(clamp(Number(midpointInput.value), midpointMin, midpointMax));
          queryInput.value = String(clamp(Number(queryInput.value), frameMin, frameMax));
        };

        const updateCostChart = () => {
          const segmentLength = clamp(Number(segmentLengthInput.value), 10, 100);
          segmentLengthInput.value = String(segmentLength);
          updateBounds(segmentLength);

          const midpoint = Number(midpointInput.value);
          const windowSize = computeWindow(segmentLength);
          const sigma = Math.max(computeSigma(segmentLength), 0.0001);
          const queryFrame = Number(queryInput.value);

          segmentLengthValue.textContent = formatFrame(segmentLength);
          midpointValue.textContent = formatFrame(midpoint);
          windowDerivedValue.textContent = formatMetric(windowSize, 2);
          windowReadout.textContent = formatMetric(windowSize, 2);
          queryValue.textContent = formatFrame(queryFrame);
          sigmaDerivedValue.textContent = sigma.toFixed(3);
          sigmaReadout.textContent = sigma.toFixed(3);

          const segmentStart = midpoint - segmentLength / 2;
          const segmentEnd = midpoint + segmentLength / 2;
          const segmentStartPx = xToPx(segmentStart);
          const segmentEndPx = xToPx(segmentEnd);
          segmentZone.setAttribute('x', segmentStartPx);
          segmentZone.setAttribute('y', padding.top);
          segmentZone.setAttribute('width', Math.max(0, segmentEndPx - segmentStartPx));
          segmentZone.setAttribute('height', plotHeight);
          segmentInterval.textContent = `[${formatFrame(segmentStart)}, ${formatFrame(segmentEnd)}]`;

          let pathData = '';
          for (let frame = frameMin; frame <= frameMax; frame += 1) {
            const point = computePoint(frame, midpoint, windowSize, sigma);
            const x = xToPx(frame);
            const y = yToPx(point.attention);
            pathData += `${frame === frameMin ? 'M' : ' L'} ${x} ${y}`;
          }
          attentionPath.setAttribute('d', pathData);

          const zoneStart = clamp(midpoint - windowSize, frameMin, frameMax);
          const zoneEnd = clamp(midpoint + windowSize, frameMin, frameMax);
          const zoneStartPx = xToPx(zoneStart);
          const zoneEndPx = xToPx(zoneEnd);
          noPenaltyZone.setAttribute('x', zoneStartPx);
          noPenaltyZone.setAttribute('y', padding.top);
          noPenaltyZone.setAttribute('width', Math.max(0, zoneEndPx - zoneStartPx));
          noPenaltyZone.setAttribute('height', plotHeight);
          zeroInterval.textContent = `[${formatFrame(zoneStart)}, ${formatFrame(zoneEnd)}]`;

          const selected = computePoint(queryFrame, midpoint, windowSize, sigma);
          const queryX = xToPx(queryFrame);
          const queryY = yToPx(selected.attention);
          queryLine.setAttribute('x1', queryX);
          queryLine.setAttribute('y1', padding.top);
          queryLine.setAttribute('x2', queryX);
          queryLine.setAttribute('y2', padding.top + plotHeight);
          queryPoint.setAttribute('cx', queryX);
          queryPoint.setAttribute('cy', queryY);
          queryLabel.setAttribute('x', Math.min(chartWidth - padding.right - 36, queryX + 8));
          queryLabel.setAttribute('y', Math.max(padding.top + 12, queryY - 8));
          queryLabel.textContent = `${(selected.attention * 100).toFixed(1)}%`;

          selectedCost.textContent = selected.cost.toFixed(3);
          selectedAttention.textContent = `${(selected.attention * 100).toFixed(1)}%`;
        };

        [segmentLengthInput, midpointInput, queryInput].forEach((input) => {
          input.addEventListener('input', updateCostChart);
        });

        drawGrid();
        updateCostChart();
      }
    }

    const overlay = document.getElementById('prompt-overlay');
    if (!overlay) {
      return;
    }

    const promptText = document.getElementById('prompt-text');
    const closeButton = overlay.querySelector('.prompt-close');
    if (!promptText || !closeButton) {
      return;
    }

    const formatPromptData = (data) => {
      if (!data || typeof data !== 'object') {
        return String(data || 'Prompt not available yet.');
      }

      const formatValue = (value, indent) => {
        const pad = ' '.repeat(indent);

        if (Array.isArray(value)) {
          return value
            .map((item, index) => `${pad}${index + 1}. ${item}`)
            .join('\n');
        }

        if (value && typeof value === 'object') {
          return Object.entries(value)
            .map(([key, nested]) => `${pad}${key}:\n${formatValue(nested, indent + 2)}`)
            .join('\n');
        }

        return `${pad}${value}`;
      };

      return Object.entries(data)
        .map(([key, value]) => {
          if (value && typeof value === 'object') {
            return `${key}:\n${formatValue(value, 2)}`;
          }
          return `${key}: ${value}`;
        })
        .join('\n\n');
    };

    const getPromptForCard = (card) => {
      if (!card) {
        return '';
      }

      const promptData = card.querySelector('.prompt-data');
      if (promptData) {
        const raw = promptData.textContent.trim();
        if (!raw) {
          return '';
        }
        try {
          return formatPromptData(JSON.parse(raw));
        } catch (error) {
          console.warn('Invalid prompt JSON', error);
          return raw;
        }
      }

      return card.dataset.prompt || '';
    };

    const openPrompt = (prompt) => {
      promptText.textContent = prompt || 'Prompt not available yet.';
      overlay.classList.add('is-visible');
      overlay.setAttribute('aria-hidden', 'false');
      document.body.classList.add('is-prompt-open');
    };

    const closePrompt = () => {
      overlay.classList.remove('is-visible');
      overlay.setAttribute('aria-hidden', 'true');
      document.body.classList.remove('is-prompt-open');
    };

    document.querySelectorAll('.prompt-button').forEach((button) => {
      button.addEventListener('click', (event) => {
        const card = event.currentTarget.closest('.video-card');
        const prompt = getPromptForCard(card);
        openPrompt(prompt);
      });
    });

    closeButton.addEventListener('click', closePrompt);
    overlay.addEventListener('click', (event) => {
      if (event.target === overlay) {
        closePrompt();
      }
    });
    document.addEventListener('keydown', (event) => {
      if (event.key === 'Escape' && overlay.classList.contains('is-visible')) {
        closePrompt();
      }
    });
})
