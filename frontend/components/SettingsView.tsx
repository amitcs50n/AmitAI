"use client";

import type { ConnectionState, UiPreferences } from "@/lib/types";
import { SettingRow, SettingsSection, Toggle } from "@/components/SettingsControls";

interface SettingsViewProps {
  connection: ConnectionState;
  preferences: UiPreferences;
  onChange: (patch: Partial<UiPreferences>) => void;
}

export function SettingsView({ connection, preferences, onChange }: SettingsViewProps) {
  return (
    <div className="min-h-0 flex-1 overflow-y-auto">
      <div className="mx-auto w-full max-w-3xl px-6 py-10 sm:px-10">
        <div className="mb-9">
          <h2 className="font-serif text-4xl text-[#eee8e1]">Settings</h2>
          <p className="mt-2 text-sm leading-6 text-[#8c857f]">Aevon interface and local development preferences.</p>
        </div>

        <SettingsSection title="General">
          <SettingRow label="Appearance" description="Aevon frontend v1 uses its private dark visual system." value="Dark" />
          <SettingRow
            label="Timestamps"
            description="Show message times in conversation history."
            value={
              <Toggle
                checked={preferences.showTimestamps}
                label="Show timestamps"
                onChange={(showTimestamps) => onChange({ showTimestamps })}
              />
            }
          />
          <SettingRow
            label="Enter to send"
            description="When disabled, use Ctrl or Command + Enter."
            value={
              <Toggle
                checked={preferences.enterToSend}
                label="Enter to send"
                onChange={(enterToSend) => onChange({ enterToSend })}
              />
            }
          />
        </SettingsSection>

        <SettingsSection title="Model" description="The browser uses only the stable AmitAI API contract.">
          <SettingRow label="Connection state" value={connection === "connected" ? "Connected" : connection === "connecting" ? "Connecting…" : "Disconnected"} />
          <SettingRow label="Endpoint" value="/api" />
        </SettingsSection>

        <SettingsSection title="Memory">
          <SettingRow label="Memory" description="Runtime memory integration is intentionally deferred." value="Not configured" />
          <SettingRow label="Manage Memory" description="Management controls will appear when real memory support exists." value="Coming later" />
        </SettingsSection>

        <SettingsSection title="Tools">
          <SettingRow label="Web" value="Not configured" />
          <SettingRow label="Files" value="Not configured" />
          <SettingRow label="Python" value="Not configured" />
          <SettingRow label="GitHub" value="Not configured" />
        </SettingsSection>

        <SettingsSection title="Developer">
          <SettingRow
            label="Developer mode"
            description="Keep technical controls available without placing them in the chat composer."
            value={
              <Toggle
                checked={preferences.developerMode}
                label="Developer mode"
                onChange={(developerMode) => onChange({ developerMode })}
              />
            }
          />
          <SettingRow
            label="Tool activity"
            description="Display real tool events when the backend supplies them."
            value={
              <Toggle
                checked={preferences.showToolActivity}
                label="Show tool activity"
                onChange={(showToolActivity) => onChange({ showToolActivity })}
              />
            }
          />
          <SettingRow
            label="Validator details"
            description="Show backend validator metadata in developer details."
            value={
              <Toggle
                checked={preferences.showValidatorDetails}
                label="Show validator details"
                onChange={(showValidatorDetails) => onChange({ showValidatorDetails })}
              />
            }
          />
          <SettingRow label="Token usage" value="Not available" />
        </SettingsSection>
      </div>
    </div>
  );
}
