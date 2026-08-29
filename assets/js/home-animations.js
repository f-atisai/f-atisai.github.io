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

  const heroTl = gsap.timeline({
    defaults: {
      ease,
      duration: 1,
    },
  });

  heroTl
    // .from(".site-header__brand", {
    //   y: -16,
    //   opacity: 0,
    // })
    // .from(
    //   ".site-header__nav a",
    //   {
    //     y: -16,
    //     opacity: 0,
    //     stagger: 0.06,
    //   },
    //   "-=0.7",
    // )
    .from(
      ".grid-cell",
      {
        opacity: 0,
        scale: 0.94,
        filter: "blur(10px)",
        stagger: {
          amount: 0.8,
          from: "random",
        },
      },
      "-=0.3",
    )
    .from(
      ".hero-meta",
      {
        y: 24,
        opacity: 0,
        stagger: 0.1,
      },
      "-=0.5",
    );

  pinnedTypingReveal({
    trigger: ".hero",
    chars: ".hero-title .char",
    end: "+=140%",
    stagger: 0.035,
  });

  gsap.to(".noise-overlay", {
    opacity: 0.075,
    backgroundPosition: "24px 18px, -18px 12px",
    duration: 6,
    ease: "sine.inOut",
    repeat: -1,
    yoyo: true,
  });

  gsap.to(".grid-layer", {
    yPercent: -8,
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
