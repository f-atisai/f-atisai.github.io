gsap.registerPlugin(ScrollTrigger, ScrollSmoother);

const reduceMotion = window.matchMedia(
  "(prefers-reduced-motion: reduce)",
).matches;
const useSmoothScroll = window.matchMedia(
  "(hover: hover) and (pointer: fine)",
).matches;

if (!reduceMotion) {
  let smoother = null;

  // Keep native scrolling on touch devices to avoid transform-heavy mobile scrolling.
  if (useSmoothScroll) {
    smoother = ScrollSmoother.create({
      wrapper: "#smooth-wrapper",
      content: "#smooth-content",
      smooth: 1.2,
    });
  }

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
      if (smoother) {
        smoother.scrollTo(target, true);
      } else {
        target.scrollIntoView({ behavior: "smooth" });
      }
    }
  });
}
