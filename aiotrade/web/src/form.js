/**
 * Formulaire de contact : case de consentement non pré-cochée, validation
 * accessible, envoi en HTTPS vers l'API (aucune donnée envoyée à un tiers ni
 * à l'outil de mesure d'audience).
 */
import { track } from './analytics.js';

const MESSAGES = {
  name: 'Indiquez votre nom.',
  email: 'Indiquez une adresse e-mail valide.',
  consent: 'Cochez la case pour nous autoriser à traiter votre demande.',
};

function setError(input, message) {
  const id = `${input.id}-error`;
  let el = document.getElementById(id);
  if (!message) {
    el?.remove();
    input.removeAttribute('aria-invalid');
    input.setAttribute('aria-describedby', (input.getAttribute('aria-describedby') || '').replace(id, '').trim());
    if (!input.getAttribute('aria-describedby')) input.removeAttribute('aria-describedby');
    return;
  }
  if (!el) {
    el = document.createElement('p');
    el.id = id;
    el.className = 'field-error';
    input.closest('.field, .field-check').append(el);
  }
  el.textContent = message;
  input.setAttribute('aria-invalid', 'true');
  const described = new Set((input.getAttribute('aria-describedby') || '').split(' ').filter(Boolean));
  described.add(id);
  input.setAttribute('aria-describedby', [...described].join(' '));
}

export function initContactForm() {
  const form = document.getElementById('contact-form');
  if (!form) return;
  const status = document.getElementById('cf-status');
  const button = form.querySelector('button[type="submit"]');

  function validate() {
    const { name, email, consent } = form.elements;
    const errors = [];
    const check = (input, ok, key) => {
      setError(input, ok ? '' : MESSAGES[key]);
      if (!ok) errors.push(input);
    };
    check(name, name.value.trim().length > 0, 'name');
    check(email, /^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(email.value.trim()), 'email');
    check(consent, consent.checked, 'consent');
    return errors;
  }

  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    const errors = validate();
    if (errors.length) {
      status.textContent = 'Le formulaire contient des erreurs.';
      errors[0].focus();
      return;
    }
    const f = form.elements;
    const payload = {
      name: f.name.value.trim(),
      email: f.email.value.trim(),
      company: f.company.value.trim(),
      message: f.message.value.trim(),
      lead_magnet: f.lead_magnet.checked,
      consent: f.consent.checked,
    };
    button.disabled = true;
    status.textContent = 'Envoi en cours…';
    try {
      const res = await fetch('/api/contact', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (res.status === 429) throw new Error('rate');
      if (!res.ok) throw new Error('http');
      form.reset();
      status.textContent = 'Merci, votre demande a bien été reçue. Nous revenons vers vous rapidement.';
      track('lead', { fiche_technique: payload.lead_magnet ? 'oui' : 'non' }); // aucune donnée personnelle
    } catch (error) {
      status.textContent =
        error.message === 'rate'
          ? 'Trop de demandes envoyées. Merci de réessayer dans quelques minutes.'
          : "L'envoi a échoué. Vérifiez votre connexion ou réessayez plus tard.";
    } finally {
      button.disabled = false;
    }
  });

  form.addEventListener('change', (event) => {
    if (event.target.getAttribute('aria-invalid') === 'true') validate();
  });
}
