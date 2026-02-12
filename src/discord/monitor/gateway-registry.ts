import type { Client } from "@buape/carbon";
import type { GatewayPlugin } from "@buape/carbon/gateway";

/**
 * Module-level registry of active Discord GatewayPlugin and Client instances.
 * Bridges the gap between agent tool handlers (which only have REST access)
 * and the gateway WebSocket (needed for operations like updatePresence).
 * Follows the same pattern as presence-cache.ts.
 */

const getGlobalMap = (key: string): Map<string, any> => {
  const g = globalThis as any;
  if (!g[key]) {
    g[key] = new Map();
  }
  return g[key];
};

const gatewayRegistry = getGlobalMap("__openclaw_discord_gateway_registry__");
const clientRegistry = getGlobalMap("__openclaw_discord_client_registry__");

// Sentinel key for the default (unnamed) account. Uses a prefix that cannot
// collide with user-configured account IDs.
const DEFAULT_ACCOUNT_KEY = "\0__default__";

function resolveAccountKey(accountId?: string): string {
  if (!accountId || accountId === "default") {
    return DEFAULT_ACCOUNT_KEY;
  }
  return accountId;
}

/** Register a GatewayPlugin instance for an account. */
export function registerGateway(accountId: string | undefined, gateway: GatewayPlugin): void {
  gatewayRegistry.set(resolveAccountKey(accountId), gateway);
}

/** Register a Client instance for an account. */
export function registerClient(accountId: string | undefined, client: Client): void {
  const key = resolveAccountKey(accountId);
  console.log(`[Registry] Registering client for account: ${accountId} (key: ${key})`);
  clientRegistry.set(key, client);
}

/** Unregister all instances for an account. */
export function unregisterGateway(accountId?: string): void {
  const key = resolveAccountKey(accountId);
  gatewayRegistry.delete(key);
  clientRegistry.delete(key);
}

/** Get the GatewayPlugin for an account. Returns undefined if not registered. */
export function getGateway(accountId?: string): GatewayPlugin | undefined {
  return gatewayRegistry.get(resolveAccountKey(accountId));
}

/** Get the Client for an account. Returns undefined if not registered. */
export function getClient(accountId?: string): Client | undefined {
  const key = resolveAccountKey(accountId);
  const client = clientRegistry.get(key);
  console.log(
    `[Registry] Get client for account: ${accountId} (key: ${key}) -> ${client ? "Found" : "Not Found"}`,
  );
  return client;
}

/** Clear all registered instances (for testing). */
export function clearGateways(): void {
  gatewayRegistry.clear();
  clientRegistry.clear();
}
