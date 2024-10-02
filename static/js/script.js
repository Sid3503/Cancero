// JavaScript for interactivity
document.addEventListener("DOMContentLoaded", function() {
    // Handle file input
    const fileInputs = document.querySelectorAll('input[type="file"]');
    fileInputs.forEach(input => {
        input.addEventListener('change', (e) => {
            const fileName = e.target.files[0].name;
            e.target.closest('.form-group').querySelector('label').innerText = `File Selected: ${fileName}`;
        });
    });

    // Confirm form submission
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', (e) => {
            alert('File is being uploaded...');
        });
    });
});
