// themeToggle.js
function toggleTheme() {
    const themeStylesheet = document.getElementById('theme-stylesheet');
    const currentTheme = themeStylesheet.getAttribute('href');
    const lightTheme = themeStylesheet.getAttribute('data-light');
    const darkTheme = themeStylesheet.getAttribute('data-dark');
    const toggleIcon = document.getElementById('theme-toggle-icon');

    if (!themeStylesheet || !toggleIcon) {
        console.error('Theme stylesheet or toggle icon not found.');
        return;
    }

    const newTheme = currentTheme === lightTheme ? darkTheme : lightTheme;
    const newThemeName = newTheme === darkTheme ? 'dark' : 'light';

    // Update theme stylesheet
    themeStylesheet.setAttribute('href', newTheme);
    localStorage.setItem('theme', newThemeName);

    // Add or remove dark-mode class
    document.body.classList.toggle('dark-mode', newThemeName === 'dark');

    // Update toggle icon
    toggleIcon.classList.toggle('fa-moon', newThemeName === 'light');
    toggleIcon.classList.toggle('fa-sun', newThemeName === 'dark');
}

/**
 * Applies the saved theme on page load.
 */
function applySavedTheme() {
    const themeStylesheet = document.getElementById('theme-stylesheet');
    const savedTheme = localStorage.getItem('theme') || 'light';
    const lightTheme = themeStylesheet.getAttribute('data-light');
    const darkTheme = themeStylesheet.getAttribute('data-dark');
    const toggleIcon = document.getElementById('theme-toggle-icon');

    if (!themeStylesheet || !toggleIcon) {
        console.error('Theme stylesheet or toggle icon not found.');
        return;
    }

    // Set the theme based on saved preference
    const newTheme = savedTheme === 'dark' ? darkTheme : lightTheme;
    themeStylesheet.setAttribute('href', newTheme);

    // Add or remove dark-mode class
    document.body.classList.toggle('dark-mode', savedTheme === 'dark');

    // Update toggle icon
    toggleIcon.classList.toggle('fa-moon', savedTheme === 'light');
    toggleIcon.classList.toggle('fa-sun', savedTheme === 'dark');
}

// Apply saved theme on DOM content loaded
document.addEventListener('DOMContentLoaded', applySavedTheme);
