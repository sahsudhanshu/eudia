/**
 * Legacy placeholder retained for backward compatibility.
 * The Better Auth client has been removed in favour of the Flask REST backend.
 */

export const authClient = null;

export function useSession() {
  throw new Error("useSession is deprecated. Import helpers from '@/lib/auth' instead.");
}