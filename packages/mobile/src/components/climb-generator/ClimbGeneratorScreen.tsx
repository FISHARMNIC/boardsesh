import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { ActivityIndicator, Pressable, ScrollView, StyleSheet, Text, View, useWindowDimensions } from 'react-native';
import { useRouter } from 'expo-router';
import type { BoardName } from '@boardsesh/shared-schema';
import { BoardImageNative } from '../BoardImageNative';
import { BleLightbulbButton } from '../ble/BleLightbulbButton';
import { trackBoardConnectTapped } from '../../lib/analytics-board-connect';
import { getBoardRenderData } from '../../lib/board-details';
import { climbGeneratorTags, generateClimbs, type GeneratedClimb } from '../../lib/climb-generator/generate';
import { hapticLight } from '../../lib/haptics';
import { useOptionalBluetoothContext } from '../../providers/bluetooth-provider';
import { useTheme } from '../../providers/theme-provider';

import { climbGeneratorModelKey } from '../../lib/climb-generator/models/supported';

export type GeneratorBoard = {
  boardName: BoardName;
  layoutId: number;
  sizeId: number;
  setIds: string;
  angle: number;
};

/** "V9–V11", or "V10" when the grader is sure. A V-grade always keeps its number. */
function gradeRangeLabel(climb: GeneratedClimb): string {
  return climb.lowV === climb.highV ? `V${climb.lowV}` : `V${climb.lowV}–V${climb.highV}`;
}
const MAX_V = 15;
/** Six candidates: a 2-column grid of 3 rows under the board. */
const RESULT_COUNT = 6;

/**
 * Experimental: generate Kilter climbs on the phone from a grade, an angle and
 * a few keywords, preview them on the board and light them, then hand the one
 * you pick back to the climb editor that opened this screen. Experimental, so
 * the copy is English (i18n-ignored).
 */
export function ClimbGeneratorScreen({ board }: { board: GeneratorBoard }) {
  const { systemColors, spacing, borderRadius, textStyles } = useTheme();
  const { t: tSettings } = useTranslation('settings');
  const { t: tCommon } = useTranslation('common');
  const { width } = useWindowDimensions();
  const router = useRouter();
  const bluetooth = useOptionalBluetoothContext();

  const [vGrade, setVGrade] = useState(4);
  const [angleOverride, setAngleOverride] = useState<number | null>(null);
  const [selectedTags, setSelectedTags] = useState<string[]>([]);
  const [results, setResults] = useState<GeneratedClimb[]>([]);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const angle = angleOverride ?? board.angle;
  const modelKey = climbGeneratorModelKey(board.boardName, board.layoutId);
  const supported = modelKey !== null;

  const renderData = useMemo(() => {
    if (!supported) return null;
    return getBoardRenderData({
      boardName: board.boardName,
      layoutId: board.layoutId,
      sizeId: board.sizeId,
      setIds: board.setIds.split(',').map(Number).filter(Boolean),
    });
  }, [board.boardName, board.layoutId, board.sizeId, board.setIds, supported]);

  const allowedPlacementIds = useMemo(
    () => new Set((renderData?.holdsData ?? []).map((hold) => hold.id)),
    [renderData],
  );
  const tags = useMemo(() => (modelKey ? climbGeneratorTags(modelKey) : []), [modelKey]);

  const toggleTag = useCallback((tag: string) => {
    hapticLight();
    setSelectedTags((current) =>
      current.includes(tag) ? current.filter((existing) => existing !== tag) : [...current, tag],
    );
  }, []);

  const generate = useCallback(async () => {
    if (!modelKey) return;
    hapticLight();
    setGenerating(true);
    setError(null);
    try {
      const climbs = await generateClimbs({
        modelKey,
        angle,
        vGrade,
        tags: selectedTags,
        allowedPlacementIds,
        keep: RESULT_COUNT,
      });
      setResults(climbs);
      setSelectedIndex(0);
      // i18n-ignore-next-line — tester-only screen
      if (climbs.length === 0) setError('No climb came out. Try fewer keywords.');
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setGenerating(false);
    }
  }, [modelKey, angle, vGrade, selectedTags, allowedPlacementIds]);

  const selected = results[selectedIndex] ?? null;
  const isConnected = bluetooth?.isConnected ?? false;
  const lightFrames = useCallback(
    (frames: string) => {
      if (!bluetooth) return;
      bluetooth.invalidateWallState();
      void bluetooth.sendFramesToBoard(frames);
    },
    [bluetooth],
  );

  // Like the main climb view's bulb: a tap connects and lights the climb on
  // screen, or re-lights it when already connected; a long press disconnects.
  const onBulbPress = useCallback(() => {
    if (!bluetooth || bluetooth.loading) return;
    if (bluetooth.isConnected) {
      if (selected) lightFrames(selected.frames);
      return;
    }
    trackBoardConnectTapped({ surface: 'climb_generator', boardName: bluetooth.boardName, reconnect: false });
    void bluetooth.connect(selected?.frames ?? '');
  }, [bluetooth, selected, lightFrames]);
  const onBulbLongPress = useCallback(() => {
    if (bluetooth?.isConnected) void bluetooth.disconnect();
  }, [bluetooth]);

  // While connected, the wall follows the candidate on screen.
  const litFramesRef = useRef<string | null>(null);
  useEffect(() => {
    if (!isConnected || !selected || litFramesRef.current === selected.frames) return;
    litFramesRef.current = selected.frames;
    lightFrames(selected.frames);
  }, [isConnected, selected, lightFrames]);
  useEffect(() => {
    if (!isConnected) litFramesRef.current = null;
  }, [isConnected]);

  // Back to the editor that opened this screen, with the chosen holds loaded
  // as a new climb (the remix path). The editor keeps its earlier draft.
  const chooseClimb = useCallback(() => {
    if (!selected) return;
    hapticLight();
    router.dismissTo({
      pathname: '/(tabs)/climbs/create',
      params: {
        forkFrames: selected.frames,
        // i18n-ignore-next-line — goes into the climb's own description, which the climber can edit
        forkDescription: `Made with the experimental Boardsesh climb generator, which graded it ${gradeRangeLabel(selected)}.`,
        ...(selectedTags.includes('no-match') ? { forkCharacteristics: JSON.stringify(['no_match']) } : {}),
        boardName: board.boardName,
        layoutId: String(board.layoutId),
        sizeId: String(board.sizeId),
        setIds: board.setIds,
        angle: String(angle),
      },
    });
  }, [selected, router, selectedTags, board.boardName, board.layoutId, board.sizeId, board.setIds, angle]);

  const styles = useMemo(
    () =>
      StyleSheet.create({
        page: { backgroundColor: systemColors.groupedBackground },
        content: { padding: spacing[4], gap: spacing[4] },
        card: {
          backgroundColor: systemColors.secondaryGroupedBackground,
          borderRadius: borderRadius.lg,
          padding: spacing[3],
          gap: spacing[3],
        },
        label: { ...textStyles.body, color: systemColors.label },
        caption: { ...textStyles.footnote, color: systemColors.secondaryLabel },
        row: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
        stepper: { flexDirection: 'row', alignItems: 'center', gap: spacing[3] },
        stepButton: {
          width: 36,
          height: 36,
          borderRadius: borderRadius.full,
          backgroundColor: systemColors.fill,
          alignItems: 'center',
          justifyContent: 'center',
        },
        stepValue: { ...textStyles.headline, color: systemColors.label, minWidth: 44, textAlign: 'center' },
        chips: { flexDirection: 'row', flexWrap: 'wrap', gap: spacing[2] },
        chip: {
          paddingHorizontal: spacing[3],
          paddingVertical: spacing[1] + 2,
          borderRadius: borderRadius.full,
          backgroundColor: systemColors.fill,
        },
        chipSelected: { backgroundColor: systemColors.accent },
        resultGrid: { flexDirection: 'row', flexWrap: 'wrap', justifyContent: 'space-between', rowGap: spacing[2] },
        resultCell: {
          // Two per row: half the card minus half the column gap.
          width: '48.5%',
          paddingVertical: spacing[2],
          borderRadius: borderRadius.lg,
          backgroundColor: systemColors.fill,
          alignItems: 'center',
        },
        resultGrade: { ...textStyles.headline, color: systemColors.label },
        resultDetail: { ...textStyles.footnote, color: systemColors.secondaryLabel },
        chipText: { ...textStyles.subheadline, color: systemColors.label },
        chipTextSelected: { color: '#FFFFFF' },
        primaryButton: {
          backgroundColor: systemColors.accent,
          borderRadius: borderRadius.lg,
          paddingVertical: spacing[3],
          alignItems: 'center',
        },
        secondaryButton: {
          backgroundColor: systemColors.fill,
          borderRadius: borderRadius.lg,
          paddingVertical: spacing[3],
          alignItems: 'center',
        },
        primaryText: { ...textStyles.headline, color: '#FFFFFF' },
        secondaryText: { ...textStyles.headline, color: systemColors.label },
        disabled: { opacity: 0.4 },
        actions: { flexDirection: 'row', alignItems: 'center', gap: spacing[3] },
        grow: { flex: 1 },
        error: { ...textStyles.footnote, color: systemColors.error },
      }),
    [systemColors, spacing, borderRadius, textStyles],
  );

  if (!supported || !renderData) {
    return (
      <ScrollView style={styles.page} contentContainerStyle={styles.content} contentInsetAdjustmentBehavior="automatic">
        <View style={styles.card}>
          {/* i18n-ignore-next-line — tester-only screen */}
          <Text style={styles.label}>There is no climb generator for this board yet.</Text>
          {/* i18n-ignore-next-line — tester-only screen */}
          <Text style={styles.caption}>
            It knows the Kilter Original, Tension Board 1, and Tension Board 2 Mirror and Spray.
          </Text>
        </View>
      </ScrollView>
    );
  }

  const boardRenderWidth = width - spacing[4] * 2 - spacing[3] * 2;

  return (
    <ScrollView style={styles.page} contentContainerStyle={styles.content} contentInsetAdjustmentBehavior="automatic">
      <View style={styles.card}>
        <BoardImageNative
          frames={selected?.frames ?? ''}
          boardName={board.boardName}
          layoutId={board.layoutId}
          sizeId={board.sizeId}
          setIds={board.setIds}
          boardWidth={renderData.boardWidth}
          boardHeight={renderData.boardHeight}
          renderWidth={boardRenderWidth}
          recyclingKey={selected?.frames ?? 'empty'}
        />
        {results.length > 0 ? (
          <View style={styles.resultGrid}>
            {results.map((climb, index) => {
              const isSelected = index === selectedIndex;
              return (
                <Pressable
                  key={climb.frames}
                  onPress={() => setSelectedIndex(index)}
                  accessibilityRole="button"
                  accessibilityState={{ selected: isSelected }}
                  style={[styles.resultCell, isSelected && styles.chipSelected]}
                >
                  {/* The range on its own line, so a V-grade is never cut at half width. */}
                  <Text style={[styles.resultGrade, isSelected && styles.chipTextSelected]}>
                    {gradeRangeLabel(climb)}
                  </Text>
                  <Text style={[styles.resultDetail, isSelected && styles.chipTextSelected]}>
                    {/* i18n-ignore-next-line — experimental screen */}
                    {`#${index + 1} · ${climb.holdCount} holds`}
                  </Text>
                </Pressable>
              );
            })}
          </View>
        ) : (
          // i18n-ignore-next-line — tester-only screen
          <Text style={styles.caption}>Pick a grade and some keywords, then generate.</Text>
        )}
        {selected ? (
          <View style={styles.actions}>
            <BleLightbulbButton
              isConnected={isConnected}
              isScanning={bluetooth?.loading ?? false}
              onPress={onBulbPress}
              onLongPress={onBulbLongPress}
              accessibilityLabel={isConnected ? tCommon('lightControl.disconnect') : tSettings('ble.connectBoard')}
              scanningAccessibilityHint={tSettings('ble.scanning')}
              writingAccessibilityHint={tSettings('ble.writing')}
            />
            <Pressable onPress={chooseClimb} style={[styles.primaryButton, styles.grow]}>
              {/* i18n-ignore-next-line — experimental screen */}
              <Text style={styles.primaryText}>Use this climb</Text>
            </Pressable>
          </View>
        ) : null}
        {/* The one generate button, always under the board. Before the first run
            it is the screen's main action; after it, "Use this climb" is, and this
            reruns with the same grade, angle and keywords. */}
        <Pressable
          onPress={generate}
          disabled={generating}
          style={[selected ? styles.secondaryButton : styles.primaryButton, generating && styles.disabled]}
        >
          {generating ? (
            <ActivityIndicator color={selected ? systemColors.label : '#FFFFFF'} />
          ) : (
            <Text style={selected ? styles.secondaryText : styles.primaryText}>
              {/* i18n-ignore-next-line — experimental screen */}
              {selected ? 'Generate new climbs' : 'Generate climbs'}
            </Text>
          )}
        </Pressable>
        {error ? <Text style={styles.error}>{error}</Text> : null}
      </View>

      <View style={styles.card}>
        <View style={styles.row}>
          {/* i18n-ignore-next-line — tester-only screen */}
          <Text style={styles.label}>Grade</Text>
          <View style={styles.stepper}>
            <Pressable style={styles.stepButton} onPress={() => setVGrade((grade) => Math.max(0, grade - 1))}>
              <Text style={styles.label}>−</Text>
            </Pressable>
            <Text style={styles.stepValue}>{`V${vGrade}`}</Text>
            <Pressable style={styles.stepButton} onPress={() => setVGrade((grade) => Math.min(MAX_V, grade + 1))}>
              <Text style={styles.label}>+</Text>
            </Pressable>
          </View>
        </View>
        <View style={styles.row}>
          {/* i18n-ignore-next-line — tester-only screen */}
          <Text style={styles.label}>Angle</Text>
          <View style={styles.stepper}>
            <Pressable style={styles.stepButton} onPress={() => setAngleOverride(Math.max(0, angle - 5))}>
              <Text style={styles.label}>−</Text>
            </Pressable>
            <Text style={styles.stepValue}>{`${angle}°`}</Text>
            <Pressable style={styles.stepButton} onPress={() => setAngleOverride(Math.min(70, angle + 5))}>
              <Text style={styles.label}>+</Text>
            </Pressable>
          </View>
        </View>
      </View>

      <View style={styles.card}>
        {/* i18n-ignore-next-line — tester-only screen */}
        <Text style={styles.label}>Keywords</Text>
        <View style={styles.chips}>
          {tags.map((tag) => {
            const isSelected = selectedTags.includes(tag);
            return (
              <Pressable
                key={tag}
                onPress={() => toggleTag(tag)}
                style={[styles.chip, isSelected && styles.chipSelected]}
              >
                <Text style={[styles.chipText, isSelected && styles.chipTextSelected]}>{tag}</Text>
              </Pressable>
            );
          })}
        </View>
        {/* i18n-ignore-next-line — tester-only screen */}
        <Text style={styles.caption}>
          Experimental model trained on public Kilter climbs. Each grade is a range the true grade should fall in 4
          times in 5.
        </Text>
      </View>
    </ScrollView>
  );
}
