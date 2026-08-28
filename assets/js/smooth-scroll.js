gsap.registerPlugin(ScrollTrigger, ScrollSmoother);

const reduceMotion = window.matchMedia(
  "(prefers-reduced-motion: reduce)",
).matches;

if (!reduceMotion) {
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
}
