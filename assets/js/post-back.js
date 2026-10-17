(() => {
  const backLink = document.querySelector("[data-post-back]");
  if (!backLink || !document.referrer) return;

  let referrer;
  try {
    referrer = new URL(document.referrer);
  } catch {
    return;
  }

  if (referrer.origin !== window.location.origin) return;

  const normalizedPath = referrer.pathname.replace(/\/+$/, "") || "/";
  const isHome =
    normalizedPath === "/" || normalizedPath.endsWith("/index.html");
  const isWork = normalizedPath.endsWith("/work.html");

  if (!isHome && !isWork) return;

  const sourceLabel = isHome ? "home" : "work";
  backLink.href = `${referrer.pathname}${referrer.search}${referrer.hash}`;
  backLink.setAttribute("aria-label", `Back to ${sourceLabel}`);

  backLink.addEventListener("click", (event) => {
    if (
      event.button !== 0 ||
      event.metaKey ||
      event.ctrlKey ||
      event.shiftKey ||
      event.altKey ||
      window.history.length <= 1
    ) {
      return;
    }

    event.preventDefault();
    window.history.back();
  });
})();
