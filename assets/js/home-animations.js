gsap.registerPlugin(ScrollTrigger, ScrollSmoother);

const reduceMotion = window.matchMedia(
  "(prefers-reduced-motion: reduce)",
).matches;

if (!reduceMotion) {
  const ease = "power4.out";

  ScrollSmoother.create({
    wrapper: "#smooth-wrapper",
    content: "#smooth-content",
    smooth: 1.2,
    effects: true,
    normalizeScroll: true,
  });

  /*
  ================================
  Anchor Link Navigation
  ================================
  */

  // Handle anchor links navigation with ScrollSmoother
  document.addEventListener("click", (e) => {
    const link = e.target.closest("a[href^='#']");
    if (!link) return;

    const targetId = link.getAttribute("href").slice(1);
    const target = document.getElementById(targetId);

    if (target) {
      e.preventDefault();
      ScrollSmoother.get().scrollTo(target, true);
    }
  });

  /*
  ================================
  Character Split
  ================================
  */

  function splitChars(selector) {
    const elements = gsap.utils.toArray(selector);

    elements.forEach((element) => {
      if (element.dataset.split === "true") return;

      const lines = element.innerHTML
        .split(/<br\s*\/?>/i)
        .map((line) => line.trim());

      element.innerHTML = lines
        .map((line) => {
          const chars = [...line]
            .map((char) => {
              if (char === " ") {
                return `<span class="char char-space">&nbsp;</span>`;
              }

              return `<span class="char">${char}</span>`;
            })
            .join("");

          return `<span class="char-line">${chars}</span>`;
        })
        .join("");

      element.dataset.split = "true";
    });
  }

  splitChars(".js-char-reveal");

  /*
  ================================
  Helpers
  ================================
  */

  function scrollyReveal({
    trigger,
    targets,
    start = "top 85%",
    end = "top 35%",
    stagger = 0.08,
    y = 80,
    rotate = 0,
  }) {
    const elements = gsap.utils.toArray(targets);

    if (!elements.length) return;

    gsap.fromTo(
      elements,
      {
        y,
        rotate,
        opacity: 0,
      },
      {
        y: 0,
        rotate: 0,
        opacity: 1,
        ease,
        stagger,
        scrollTrigger: {
          trigger,
          start,
          end,
          scrub: 1,
        },
      },
    );
  }

  function scrollyLines({
    trigger,
    lines,
    start = "top 85%",
    end = "top 35%",
    stagger = 0.08,
  }) {
    const elements = gsap.utils.toArray(lines);

    if (!elements.length) return;

    gsap.fromTo(
      elements,
      {
        yPercent: 120,
        rotate: 2,
        opacity: 0,
      },
      {
        yPercent: 0,
        rotate: 0,
        opacity: 1,
        ease,
        stagger,
        scrollTrigger: {
          trigger,
          start,
          end,
          scrub: 1,
        },
      },
    );
  }

  function pinnedTypingReveal({
    trigger,
    chars,
    start = "top top",
    end = "+=120%",
    stagger = 0.035,
    pin = true,
  }) {
    const elements = gsap.utils.toArray(chars);

    if (!elements.length) return;

    gsap.fromTo(
      elements,
      {
        opacity: 0,
      },
      {
        opacity: 1,
        ease: "none",
        stagger,
        scrollTrigger: {
          trigger,
          start,
          end,
          scrub: true,
          pin,
          anticipatePin: 1,
        },
      },
    );
  }

  /*
  ================================
  Hero: Load + Ambient
  ================================
  */

  const hero = document.querySelector(".hero");
  const heroReveal = hero ? hero.querySelector(".hero__reveal") : null;
  const heroRevealMask = hero ? hero.querySelector(".hero__reveal-mask") : null;
  const heroSmokeMask = hero ? hero.querySelector(".hero__smoke-mask") : null;
  const hasHoverPointer = window.matchMedia(
    "(hover: hover) and (pointer: fine)",
  ).matches;

  if (hero && heroReveal && heroRevealMask && heroSmokeMask) {
    // Resting size: minimum 85px, maximum 160px, otherwise 11% of the viewport width.
    const blobRadius = () =>
      Math.max(130, Math.min(160, window.innerWidth * 0.11));
    // Movement growth: 1.3 makes the blob 30% larger (use 1.1 for 10%).
    const movingBlobRadius = () => blobRadius() * 1.3;
    // Initial growth: 2 makes the centered blob grow to twice its resting size.
    const initialBlobScale = 2;
    let blobX = 0;
    let blobY = 0;
    let targetX = 0;
    let targetY = 0;
    let isPointerMoving = false;
    let idleTimer;
    let initialBlobTimeline;
    let ambientTween;
    let lastAmbientDirection = -1;

    // Resting drift distance in pixels. Increase for wider ambient movement.
    const ambientTravel = 18;
    const ambientOffset = { x: 0, y: 0 };
    const ambientDirections = [
      { x: -1, y: 0 },
      { x: 1, y: 0 },
      { x: 0, y: -1 },
      { x: 0, y: 1 },
      { x: -0.72, y: -0.72 },
      { x: 0.72, y: -0.72 },
      { x: -0.72, y: 0.72 },
      { x: 0.72, y: 0.72 },
    ];

    const setBlobRadius = (radius, duration, ease) => {
      gsap.to([heroRevealMask, heroSmokeMask], {
        attr: { r: radius },
        duration,
        ease,
        overwrite: true,
      });
    };

    const stopAmbientDrift = () => {
      if (ambientTween) ambientTween.kill();

      ambientTween = gsap.to(ambientOffset, {
        x: 0,
        y: 0,
        duration: 0.45,
        ease: "power2.out",
        overwrite: true,
      });
    };

    const startAmbientDrift = () => {
      if (isPointerMoving) return;
      if (ambientTween) ambientTween.kill();

      let directionIndex = Math.floor(Math.random() * ambientDirections.length);

      if (directionIndex === lastAmbientDirection) {
        directionIndex = (directionIndex + 1) % ambientDirections.length;
      }

      lastAmbientDirection = directionIndex;
      const direction = ambientDirections[directionIndex];

      ambientTween = gsap.to(ambientOffset, {
        x: direction.x * ambientTravel,
        y: direction.y * ambientTravel,
        // Resting drift duration in seconds. Increase both values for slower movement.
        duration: gsap.utils.random(2.0, 3.8),
        ease: "sine.inOut",
        overwrite: true,
        onComplete: startAmbientDrift,
      });
    };

    gsap.ticker.add(() => {
      // Cursor-follow smoothing: lower values trail more; higher values follow more tightly.
      blobX += (targetX - blobX) * 0.16;
      blobY += (targetY - blobY) * 0.16;
      heroRevealMask.setAttribute("cx", blobX + ambientOffset.x);
      heroRevealMask.setAttribute("cy", blobY + ambientOffset.y);
      heroSmokeMask.setAttribute("cx", blobX + ambientOffset.x);
      heroSmokeMask.setAttribute("cy", blobY + ambientOffset.y);
    });

    const cancelInitialBlobAnimation = () => {
      if (!initialBlobTimeline) return;

      initialBlobTimeline.kill();
      initialBlobTimeline = null;
    };

    const showCenteredBlob = () => {
      const bounds = hero.getBoundingClientRect();

      blobX = targetX = bounds.width / 2;
      blobY = targetY = bounds.height / 2;
      heroRevealMask.setAttribute("cx", blobX);
      heroRevealMask.setAttribute("cy", blobY);
      heroSmokeMask.setAttribute("cx", blobX);
      heroSmokeMask.setAttribute("cy", blobY);
      heroReveal.classList.add("is-visible");

      initialBlobTimeline = gsap
        .timeline({
          onComplete: () => {
            initialBlobTimeline = null;
            startAmbientDrift();
          },
        })
        // Initial growth duration in seconds. Higher values make expansion slower.
        .to([heroRevealMask, heroSmokeMask], {
          attr: { r: blobRadius() * initialBlobScale },
          duration: 1.4,
          ease: "power3.out",
        })
        // Initial return duration in seconds. Higher values make settling slower.
        .to([heroRevealMask, heroSmokeMask], {
          attr: { r: blobRadius() },
          duration: 1.1,
          ease: "power2.inOut",
        });
    };

    // Begin with the reveal centered; pointer interaction takes over on first movement.
    showCenteredBlob();

    const placeBlob = (event) => {
      cancelInitialBlobAnimation();
      stopAmbientDrift();

      const bounds = hero.getBoundingClientRect();

      targetX = event.clientX - bounds.left;
      targetY = event.clientY - bounds.top;

      if (!isPointerMoving) {
        isPointerMoving = true;
        // Growth duration in seconds: higher values make movement growth slower.
        setBlobRadius(movingBlobRadius(), 1.8, "power2.out");
      }

      window.clearTimeout(idleTimer);
      idleTimer = window.setTimeout(() => {
        isPointerMoving = false;
        // Return duration in seconds: higher values return to resting size more slowly.
        setBlobRadius(blobRadius(), 1.5, "power2.out");
        startAmbientDrift();
        // Stationary delay in milliseconds before the return animation begins.
      }, 240);
    };

    hero.addEventListener("pointerenter", (event) => {
      cancelInitialBlobAnimation();
      stopAmbientDrift();

      const bounds = hero.getBoundingClientRect();

      blobX = targetX = event.clientX - bounds.left;
      blobY = targetY = event.clientY - bounds.top;
      heroRevealMask.setAttribute("cx", blobX);
      heroRevealMask.setAttribute("cy", blobY);
      heroSmokeMask.setAttribute("cx", blobX);
      heroSmokeMask.setAttribute("cy", blobY);
      heroReveal.classList.add("is-visible");
      setBlobRadius(blobRadius(), 0.8, "elastic.out(1, 0.55)");
    });

    hero.addEventListener("pointerdown", (event) => {
      if (!hasHoverPointer) placeBlob(event);
    });

    hero.addEventListener("pointermove", placeBlob);

    hero.addEventListener("pointerleave", () => {
      window.clearTimeout(idleTimer);
      isPointerMoving = false;

      if (hasHoverPointer) {
        if (ambientTween) ambientTween.kill();
        heroReveal.classList.remove("is-visible");
        setBlobRadius(0, 0.45, "power3.in");
      } else {
        // Touch pointers have no persistent hover, so keep the reveal visible after release.
        setBlobRadius(blobRadius(), 1.5, "power2.out");
        startAmbientDrift();
      }
    });

    window.addEventListener("resize", () => {
      if (heroReveal.classList.contains("is-visible")) {
        gsap.set([heroRevealMask, heroSmokeMask], {
          attr: { r: isPointerMoving ? movingBlobRadius() : blobRadius() },
        });
      }
    });
  }

  const heroTl = gsap.timeline({
    defaults: {
      ease,
      duration: 1.1,
    },
  });

  heroTl
    .from(".hero__guides", { opacity: 0, duration: 1.8 })
    .from(
      ".hero__eyebrow > span",
      { y: 14, opacity: 0, stagger: 0.08 },
      "-=1.45",
    )
    .from(
      ".hero__word",
      {
        yPercent: 115,
        rotate: 2,
        transformOrigin: "left bottom",
        duration: 1.45,
        stagger: 0.1,
      },
      "-=1.1",
    )
    .from(".hero__summary", { y: 20, opacity: 0, duration: 0.9 }, "-=0.75")
    .from(
      ".hero__footer > *",
      { y: 12, opacity: 0, duration: 0.8, stagger: 0.08 },
      "-=0.65",
    );

  gsap.to(".hero__content", {
    yPercent: -5,
    opacity: 0.45,
    ease: "none",
    scrollTrigger: {
      trigger: ".hero",
      start: "top top",
      end: "bottom top",
      scrub: 1,
    },
  });

  /*
  ================================
  About
  ================================
  */

  scrollyReveal({
    trigger: ".home-about",
    targets: ".home-about__kicker",
  });

  scrollyLines({
    trigger: ".home-about",
    lines: ".home-about__statement .line-mask > span",
    start: "top 80%",
    end: "center 45%",
  });

  scrollyReveal({
    trigger: ".home-about",
    targets: ".home-about__copy",
    start: "top 65%",
    end: "center 35%",
    y: 100,
  });

  /*
  ================================
  Work
  ================================
  */

  scrollyReveal({
    trigger: ".home-work",
    targets: ".home-work__header > *",
    start: "top 85%",
    end: "top 50%",
  });

  gsap.utils.toArray(".work-preview").forEach((card) => {
    const eyebrow = card.querySelector(".work-preview__eyebrow");
    const titleChars = card.querySelectorAll(".work-preview__title .char");
    const image = card.querySelector(".work-preview__image");
    const imageImg = card.querySelector(".work-preview__image img");
    const description = card.querySelector(".work-preview__description");
    const title = card.querySelector(".work-preview__title");

    ScrollTrigger.create({
      trigger: card,
      start: "top top",
      end: "bottom top",

      onEnter: () => {
        card.classList.add("is-active");
      },

      onLeave: () => {
        card.classList.remove("is-active");
      },

      onEnterBack: () => {
        card.classList.add("is-active");
      },

      onLeaveBack: () => {
        card.classList.remove("is-active");
      },
    });

    scrollyReveal({
      trigger: card,
      targets: eyebrow,
      start: "top 85%",
      end: "top 55%",
      y: 50,
    });

    pinnedTypingReveal({
      trigger: card,
      chars: titleChars,
      end: "+=100%",
      stagger: 0.035,
    });

    if (image) {
      gsap.fromTo(
        image,
        {
          clipPath: "inset(0 0 100% 0)",
          rotate: -2,
        },
        {
          clipPath: "inset(0 0 0% 0)",
          rotate: 0,
          ease,
          scrollTrigger: {
            trigger: card,
            start: "top 75%",
            end: "center 35%",
            scrub: 1,
          },
        },
      );

      gsap.to(image, {
        yPercent: -5,
        ease: "none",
        scrollTrigger: {
          trigger: card,
          start: "top bottom",
          end: "bottom top",
          scrub: true,
        },
      });
    }

    if (imageImg) {
      gsap.fromTo(
        imageImg,
        {
          scale: 1.16,
          "--image-blur": "10px",
        },
        {
          scale: 1,
          "--image-blur": "0px",
          ease,
          scrollTrigger: {
            trigger: card,
            start: "top 75%",
            end: "center 35%",
            scrub: 1,
          },
        },
      );
    }

    scrollyReveal({
      trigger: card,
      targets: description,
      start: "top 60%",
      end: "center 30%",
      y: 80,
    });

    card.addEventListener("mouseenter", () => {
      if (title) {
        gsap.to(title, {
          x: 16,
          duration: 0.5,
          ease,
        });
      }

      if (imageImg) {
        gsap.to(imageImg, {
          scale: 1.04,
          duration: 0.7,
          ease,
        });
      }
    });

    card.addEventListener("mouseleave", () => {
      if (title) {
        gsap.to(title, {
          x: 0,
          duration: 0.5,
          ease,
        });
      }

      if (imageImg) {
        gsap.to(imageImg, {
          scale: 1,
          duration: 0.7,
          ease,
        });
      }
    });
  });

  /*
  ================================
  Services
  ================================
  */

  scrollyReveal({
    trigger: ".home-services",
    targets: ".home-services__kicker",
  });

  scrollyLines({
    trigger: ".home-services",
    lines: ".home-services__title .line-mask > span",
    start: "top 80%",
    end: "center 45%",
  });

  gsap.utils.toArray(".home-services__item").forEach((item) => {
    gsap.fromTo(
      item,
      {
        scaleX: 0,
        transformOrigin: "left",
      },
      {
        scaleX: 1,
        ease,
        scrollTrigger: {
          trigger: item,
          start: "top 85%",
          end: "top 55%",
          scrub: 1,
        },
      },
    );

    scrollyReveal({
      trigger: item,
      targets: item.querySelectorAll("span, p"),
      start: "top 82%",
      end: "top 52%",
      stagger: 0.1,
      y: 40,
    });
  });

  /*
  ================================
  Footer
  ================================
  */

  gsap.to(".site-footer__marquee", {
    xPercent: -50,
    duration: 24,
    ease: "none",
    repeat: -1,
  });

  pinnedTypingReveal({
    trigger: ".site-footer",
    chars: ".site-footer__statement .char",
    end: "+=120%",
    stagger: 0.03,
  });

  scrollyReveal({
    trigger: ".site-footer",
    targets: ".site-footer__nav, .site-footer__bottom",
    start: "top 65%",
    end: "center 40%",
    y: 40,
    stagger: 0.12,
  });

  /*
  ================================
  Refresh
  ================================
  */

  window.addEventListener("load", () => {
    ScrollTrigger.refresh();
  });
} else {
  gsap.set("*", { clearProps: "all" });

  if (window.ScrollTrigger) {
    ScrollTrigger.getAll().forEach((trigger) => trigger.kill());
  }
}
