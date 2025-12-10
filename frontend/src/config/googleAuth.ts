export const GOOGLE_CLIENT_ID = process.env.REACT_APP_GOOGLE_CLIENT_ID || '';

// Validate that CLIENT_ID is configured
if (!GOOGLE_CLIENT_ID || GOOGLE_CLIENT_ID === 'your-google-client-id-here.apps.googleusercontent.com') {
  console.warn(' Google OAuth Client ID is not configured. Please set REACT_APP_GOOGLE_CLIENT_ID in your .env file.');
}

export const GOOGLE_SCOPES = [
  'openid',
  'profile',
  'email'
].join(' ');

export const GOOGLE_LOGIN_CONFIG = {
  clientId: GOOGLE_CLIENT_ID,
  redirectUri: window.location.origin,
  scope: GOOGLE_SCOPES,
  responseType: 'code',
  accessType: 'offline',
  prompt: 'consent'
};

// Check if running in development
export const isDevelopment = process.env.NODE_ENV === 'development';

// Valid client ID format validation
export const isValidClientId = (clientId: string): boolean => {
  return Boolean(clientId) && 
         clientId.length > 0 && 
         clientId.includes('.apps.googleusercontent.com') &&
         clientId !== 'your-google-client-id-here.apps.googleusercontent.com';
};
