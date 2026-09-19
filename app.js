/* ============================================================
   HepaJScore v18 — client-side behaviour
   ============================================================ */

(function () {
  'use strict';

  /* ---- Mobile nav toggle ---- */
  const navToggle = document.getElementById('navToggle');
  const navLinks  = document.getElementById('navLinks');
  if (navToggle && navLinks) {
    navToggle.addEventListener('click', () => navLinks.classList.toggle('open'));
  }

  /* ---- Photo: camera + upload (two separate inputs) ---- */
  const fileInput   = document.getElementById('eyePhoto');
  const cameraInput = document.getElementById('eyeCamera');
  const zone        = document.getElementById('photoZone');
  const preview     = document.getElementById('photoPreview');
  const previewImg  = document.getElementById('previewImg');
  const fileNameEl  = document.getElementById('photoFilename');
  const retakeBtn   = document.getElementById('retakeBtn');

  function showPreview(file) {
    if (!file) return;
    const url = URL.createObjectURL(file);
    if (previewImg) previewImg.src = url;
    if (fileNameEl) fileNameEl.textContent = file.name;
    if (preview) preview.classList.add('shown');
    if (zone) { zone.classList.add('has-image'); zone.style.borderColor = ''; }
  }

  function activeFile() {
    if (cameraInput && cameraInput.files && cameraInput.files.length) return cameraInput.files[0];
    if (fileInput   && fileInput.files   && fileInput.files.length)   return fileInput.files[0];
    return null;
  }

  if (cameraInput) {
    cameraInput.addEventListener('change', () => {
      if (cameraInput.files && cameraInput.files.length) {
        if (fileInput) fileInput.value = '';
        showPreview(cameraInput.files[0]);
      }
    });
  }
  if (fileInput) {
    fileInput.addEventListener('change', () => {
      if (fileInput.files && fileInput.files.length) {
        if (cameraInput) cameraInput.value = '';
        showPreview(fileInput.files[0]);
      }
    });
  }
  if (retakeBtn) {
    retakeBtn.addEventListener('click', () => {
      if (fileInput)   fileInput.value = '';
      if (cameraInput) cameraInput.value = '';
      if (preview) preview.classList.remove('shown');
      if (zone) zone.classList.remove('has-image');
      if (previewImg) previewImg.src = '';
    });
  }

  /* ---- Submit: spinner + guard ---- */
  const form = document.getElementById('assessForm');
  const submitBtn = document.getElementById('submitBtn');
  if (form && submitBtn) {
    form.addEventListener('submit', (e) => {
      if (!activeFile()) {
        e.preventDefault();
        if (zone) {
          zone.style.borderColor = 'var(--err)';
          zone.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
        return;
      }
      submitBtn.classList.add('loading');
      submitBtn.disabled = true;
      const label = submitBtn.querySelector('.btn-submit-label');
      if (label && submitBtn.dataset.analyzing) {
        label.textContent = submitBtn.dataset.analyzing;
      }
    });
  }

  /* ---- Auto-dismiss flash messages ---- */
  document.querySelectorAll('.flash').forEach((el) => {
    if (el.classList.contains('success') || el.classList.contains('info')) {
      setTimeout(() => {
        el.style.transition = 'opacity .4s, transform .4s';
        el.style.opacity = '0';
        el.style.transform = 'translateY(-6px)';
        setTimeout(() => el.remove(), 400);
      }, 6000);
    }
  });

  /* ---- PWA: service worker registration ---- */
  if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
      navigator.serviceWorker.register('/service-worker.js').catch(() => {});
    });
  }

  /* ---- PWA: install prompt ---- */
  let deferredPrompt = null;
  const installEl  = document.getElementById('installPrompt');
  const installBtn = document.getElementById('installBtn');
  window.addEventListener('beforeinstallprompt', (e) => {
    e.preventDefault();
    deferredPrompt = e;
    if (installEl) installEl.classList.add('shown');
  });
  if (installBtn) {
    installBtn.addEventListener('click', async () => {
      if (!deferredPrompt) return;
      deferredPrompt.prompt();
      await deferredPrompt.userChoice;
      deferredPrompt = null;
      if (installEl) installEl.classList.remove('shown');
    });
  }
  window.addEventListener('appinstalled', () => {
    if (installEl) installEl.classList.remove('shown');
  });
})();
