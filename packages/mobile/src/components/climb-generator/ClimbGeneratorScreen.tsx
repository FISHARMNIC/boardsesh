import { useCallback, useMemo, useState } from 'react';
import { ActivityIndicator, Pressable, ScrollView, StyleSheet, Text, View, useWindowDimensions } from 'react-native';
import { toBoardName } from '@boardsesh/board-config';
import { BoardImageNative } from '../BoardImageNative';
import { getBoardRenderData } from '../../lib/board-details';
import { useActiveBoard } from '../../lib/graphql/use-active-board';
import { climbGeneratorTags, generateClimbs, type GeneratedClimb } from '../../lib/climb-generator/generate';
import { hapticLight } from '../../lib/haptics';
import { useOptionalBluetoothContext } from '../../providers/bluetooth-provider';
import { useTheme } from '../../providers/theme-provider';

const KILTER_ORIGINAL_LAYOUT_ID = 1;

/** "V9–V11", or "V10" when the grader is sure. A V-grade always keeps its number. */
function gradeRangeLabel(climb: GeneratedClimb): string {
  return climb.lowV === climb.highV ? `V${climb.lowV}` : `V${climb.lowV}–V${climb.highV}`;
}
const MAX_V = 15;

/**
 * Experimental: generate Kilter climbs on the phone from a grade, an angle and
 * a few keywords, preview them on the board and light them. Tester-only, so
 * the copy is English (i18n-ignored) like the other Developer tools.
 */
export function ClimbGeneratorScreen() {
  const { systemColors, spacing, borderRadius, textStyles } = useTheme();
  const { width } = useWindowDimensions();
  const { data: activeBoard } = useActiveBoard();
  const bluetooth = useOptionalBluetoothContext();

  const [vGrade, setVGrade] = useState(4);
  const [angleOverride, setAngleOverride] = useState<number | null>(null);
  const [selectedTags, setSelectedTags] = useState<string[]>([]);
  const [results, setResults] = useState<GeneratedClimb[]>([]);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const angle = angleOverride ?? activeBoard?.angle ?? 40;
  const boardName = toBoardName(activeBoard?.boardType);
  const supported = boardName === 'kilter' && activeBoard?.layoutId === KILTER_ORIGINAL_LAYOUT_ID;

  const renderData = useMemo(() => {
    if (!activeBoard || !boardName || !supported) return null;
    return getBoardRenderData({
      boardName,
      layoutId: activeBoard.layoutId,
      sizeId: activeBoard.sizeId,
      setIds: activeBoard.setIds.split(',').map(Number).filter(Boolean),
    });
  }, [activeBoard, boardName, supported]);

  const allowedPlacementIds = useMemo(
    () => new Set((renderData?.holdsData ?? []).map((hold) => hold.id)),
    [renderData],
  );
  const tags = useMemo(() => (supported ? climbGeneratorTags() : []), [supported]);

  const toggleTag = useCallback((tag: string) => {
    hapticLight();
    setSelectedTags((current) =>
      current.includes(tag) ? current.filter((existing) => existing !== tag) : [...current, tag],
    );
  }, []);

  const generate = useCallback(async () => {
    hapticLight();
    setGenerating(true);
    setError(null);
    try {
      const climbs = await generateClimbs({ angle, vGrade, tags: selectedTags, allowedPlacementIds });
      setResults(climbs);
      setSelectedIndex(0);
      // i18n-ignore-next-line — tester-only screen
      if (climbs.length === 0) setError('No climb came out. Try fewer keywords.');
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setGenerating(false);
    }
  }, [angle, vGrade, selectedTags, allowedPlacementIds]);

  const selected = results[selectedIndex] ?? null;
  const canLight = (bluetooth?.isConnected ?? false) && selected !== null;
  const lightIt = useCallback(() => {
    if (!bluetooth || !selected) return;
    hapticLight();
    bluetooth.invalidateWallState();
    void bluetooth.sendFramesToBoard(selected.frames);
  }, [bluetooth, selected]);

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
        error: { ...textStyles.footnote, color: systemColors.error },
      }),
    [systemColors, spacing, borderRadius, textStyles],
  );

  if (!supported || !renderData || !activeBoard || !boardName) {
    return (
      <ScrollView style={styles.page} contentContainerStyle={styles.content} contentInsetAdjustmentBehavior="automatic">
        <View style={styles.card}>
          {/* i18n-ignore-next-line — tester-only screen */}
          <Text style={styles.label}>The generator only knows the Kilter Board Original layout so far.</Text>
          {/* i18n-ignore-next-line — tester-only screen */}
          <Text style={styles.caption}>Switch your active board to a Kilter Original and come back.</Text>
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
          boardName={boardName}
          layoutId={activeBoard.layoutId}
          sizeId={activeBoard.sizeId}
          setIds={activeBoard.setIds}
          boardWidth={renderData.boardWidth}
          boardHeight={renderData.boardHeight}
          renderWidth={boardRenderWidth}
          recyclingKey={selected?.frames ?? 'empty'}
        />
        {results.length > 0 ? (
          <View style={styles.chips}>
            {results.map((climb, index) => (
              <Pressable
                key={climb.frames}
                onPress={() => setSelectedIndex(index)}
                style={[styles.chip, index === selectedIndex && styles.chipSelected]}
              >
                <Text style={[styles.chipText, index === selectedIndex && styles.chipTextSelected]}>
                  {/* i18n-ignore-next-line — tester-only screen */}
                  {`#${index + 1} · ${gradeRangeLabel(climb)} · ${climb.holdCount} holds`}
                </Text>
              </Pressable>
            ))}
          </View>
        ) : (
          // i18n-ignore-next-line — tester-only screen
          <Text style={styles.caption}>Pick a grade and some keywords, then generate.</Text>
        )}
        {selected ? (
          <Pressable
            onPress={lightIt}
            disabled={!canLight}
            style={[styles.secondaryButton, !canLight && styles.disabled]}
          >
            <Text style={styles.secondaryText}>
              {/* i18n-ignore-next-line — tester-only screen */}
              {canLight ? 'Light it on the board' : 'Connect a board to light it'}
            </Text>
          </Pressable>
        ) : null}
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

      <Pressable onPress={generate} disabled={generating} style={[styles.primaryButton, generating && styles.disabled]}>
        {generating ? (
          <ActivityIndicator color="#FFFFFF" />
        ) : (
          // i18n-ignore-next-line — tester-only screen
          <Text style={styles.primaryText}>Generate climbs</Text>
        )}
      </Pressable>
      {error ? <Text style={styles.error}>{error}</Text> : null}
    </ScrollView>
  );
}
