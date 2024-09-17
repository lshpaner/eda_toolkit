document.addEventListener('DOMContentLoaded', (event) => {
    const images = document.querySelectorAll('.no-click img');
    images.forEach((img) => {
        img.style.pointerEvents = 'none';
    });
});