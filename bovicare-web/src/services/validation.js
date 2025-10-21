// Simple, reusable validators for auth and forgot-password flows

// Email regex kept readable and permissive enough for most cases
export function isValidEmail(email) {
  if (typeof email !== 'string') return false;
  const trimmed = email.trim();
  if (trimmed.length === 0) return false;
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/i;
  return emailRegex.test(trimmed);
}

// Brazilian phone helper: accepts formats with spaces, dashes, parentheses, and leading +55
// Normalizes to digits and validates 10-11 digits (landline or mobile with 9th digit)
export function isValidBRPhone(phone) {
  if (typeof phone !== 'string' && typeof phone !== 'number') return false;
  const digits = String(phone).replace(/\D/g, '');
  // Strip country code 55 if provided
  const normalized = digits.startsWith('55') && digits.length > 11 ? digits.slice(2) : digits;
  return normalized.length === 10 || normalized.length === 11;
}

export function isValidSixDigitCode(code) {
  const str = String(code || '').trim();
  return /^\d{6}$/.test(str);
}

// Step 1: choose recovery method and provide email or phone
export function validateForgotPasswordSelection({ method, email, phone }) {
  const errors = {};

  if (method !== 'email' && method !== 'sms') {
    errors.method = 'Escolha como deseja recuperar o acesso';
  }

  if (method === 'email') {
    if (!isValidEmail(email)) {
      errors.email = 'Informe um e-mail válido';
    }
  }

  if (method === 'sms') {
    if (!isValidBRPhone(phone)) {
      errors.phone = 'Informe um número de telefone válido';
    }
  }

  return { isValid: Object.keys(errors).length === 0, errors };
}

// Step 2: confirmation code received via e-mail or SMS
export function validateForgotPasswordCode(code) {
  const errors = {};
  if (!isValidSixDigitCode(code)) {
    errors.code = 'Código deve conter 6 dígitos';
  }
  return { isValid: Object.keys(errors).length === 0, errors };
}

// Optional: final step to set a new password (basic constraints)
export function validateNewPassword({ password, confirmPassword, minLength = 8 }) {
  const errors = {};
  if (typeof password !== 'string' || password.length < minLength) {
    errors.password = `A senha deve ter pelo menos ${minLength} caracteres`;
  }
  if (password !== confirmPassword) {
    errors.confirmPassword = 'As senhas não coincidem';
  }
  return { isValid: Object.keys(errors).length === 0, errors };
}

export default {
  isValidEmail,
  isValidBRPhone,
  isValidSixDigitCode,
  validateForgotPasswordSelection,
  validateForgotPasswordCode,
  validateNewPassword,
};


