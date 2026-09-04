"use client";

import type { ConnectionState, InferenceMode, UiPreferences, VisionCapability } from "@/lib/types";
import { SettingRow, SettingsSection, Toggle } from "@/components/SettingsControls";

interface SettingsViewProps {
  connection: ConnectionState;
  inferenceMode?: InferenceMode;
  vision?: VisionCapability | null;
  onReloadCapabilities?: () => void;
  preferences: UiPreferences;
  onChange: (patch: Partial<UiPreferences>) => void;
}

export function SettingsView({ connection, inferenceMode = "unknown", vision, onReloadCapabilities, preferences, onChange }: SettingsViewProps) {
  const modeLabel = { mock: "Mock · no model inference", local: "Local inference", remote: "Remote inference", unknown: "Unavailable" }[inferenceMode];
  const imageLabel = !vision ? "Unavailable" : vision.enabled && vision.scope === "local" ? "Local vision" : vision.enabled && vision.scope === "remote" ? "Remote · consent required" : "Not enabled";
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

        <SettingsSection title="Model" description="Inference is selected in the server configuration. This status does not confirm that model weights are loaded.">
          <SettingRow label="Connection state" value={connection === "connected" ? "Connected" : connection === "connecting" ? "Connecting…" : "Disconnected"} />
          <SettingRow label="Configured inference" value={modeLabel} />
          <SettingRow label="Image analysis" description="One image per message. Remote analysis requires fresh consent for each send or retry." value={imageLabel} />
          <button className="mt-3 text-sm text-[#dca778]" onClick={onReloadCapabilities} type="button">Refresh capabilities</button>
        </SettingsSection>

        <SettingsSection title="Memory">
          <SettingRow label="Memory" description="Explicit structured memories are managed by the local AmitAI backend." value="Configured" />
          <SettingRow label="Manage Memory" description="Open Memory from the sidebar to search, edit, create, or forget entries." value="Sidebar" />
        </SettingsSection>

        <SettingsSection title="Tools">
          <SettingRow label="Calculator" description="The inference runtime can invoke the calculator while answering arithmetic questions." value={inferenceMode === "mock" ? "Not used in mock mode" : inferenceMode === "unknown" ? "Unavailable" : "Runtime tool"} />
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
