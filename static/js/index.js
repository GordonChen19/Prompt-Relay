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

    const overlay = document.getElementById('prompt-overlay');
    if (!overlay) {
      return;
    }

    const promptText = document.getElementById('prompt-text');
    const closeButton = overlay.querySelector('.prompt-close');

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
