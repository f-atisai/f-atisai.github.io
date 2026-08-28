const weatherEl = document.querySelector(".post-hero__location");
const datetimeEl = document.querySelector(".datetime");

// -----------------------------
// Local date & time
// -----------------------------
function updateDateTime() {
  const now = new Date();

  const year = now.getFullYear();
  const month = String(now.getMonth() + 1).padStart(2, "0");
  const day = String(now.getDate()).padStart(2, "0");

  const hours = String(now.getHours()).padStart(2, "0");
  const minutes = String(now.getMinutes()).padStart(2, "0");

  const formattedDateTime = `${year}-${month}-${day} — ${hours}:${minutes}`;

  datetimeEl.textContent = formattedDateTime;
  datetimeEl.dateTime = `${year}-${month}-${day}T${hours}:${minutes}`;
}

updateDateTime();
setInterval(updateDateTime, 30 * 1000);

// -----------------------------
// Approximate location + weather
// -----------------------------
async function updateLocationAndWeather() {
  try {
    // IP-based approximate location
    const locationResponse = await fetch("https://ipwho.is/");

    if (!locationResponse.ok) {
      throw new Error("Location lookup failed");
    }

    const location = await locationResponse.json();

    if (!location.success) {
      throw new Error("Unable to determine approximate location");
    }

    const { city, latitude, longitude } = location;

    // Current weather from Open-Meteo
    const weatherResponse = await fetch(
      `https://api.open-meteo.com/v1/forecast` +
        `?latitude=${latitude}` +
        `&longitude=${longitude}` +
        `&current=temperature_2m`,
    );

    if (!weatherResponse.ok) {
      throw new Error("Weather lookup failed");
    }

    const weather = await weatherResponse.json();

    const temperature = Math.round(weather.current.temperature_2m);

    weatherEl.textContent = `${city || "Local"} • ${temperature}°C`;
  } catch (error) {
    console.error("Location/weather error:", error);

    weatherEl.textContent = "Weather unavailable";
  }
}

updateLocationAndWeather();

// Refresh weather every 10 minutes
setInterval(updateLocationAndWeather, 10 * 60 * 1000);
